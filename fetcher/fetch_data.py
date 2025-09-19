"""
Bus Data Fetcher - Integrated with bus-easebackend.onrender.com
"""
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
import json
import time
from config.settings import settings

# Import Gemini
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

class RealTimeBusDataFetcher:
    """Bus data fetcher integrated with your backend API."""
    
    def __init__(self):
        self.session = None
        self.base_url = settings.bus_api_base_url
        self.last_fetch_time = 0
        self.cache_duration = settings.bus_data_cache_duration
        self.cached_data = None
        
        # Initialize Gemini for location conversion
        self.gemini_client = None
        self.location_cache = {}
        self.location_cache_duration = settings.gemini_location_cache_duration
        self._initialize_gemini()
        
        print(f"Bus Data Fetcher initialized")
        print(f"   Backend URL: {self.base_url}")
        print(f"   Cache Duration: {self.cache_duration}s")
    
    def _initialize_gemini(self):
        """Initialize Gemini for location conversion."""
        if not GENAI_AVAILABLE:
            print("Gemini not available - using fallback location conversion")
            return
        
        try:
            if settings.is_gemini_location_enabled():
                genai.configure(api_key=settings.gemini_api_key)
                self.gemini_client = genai.GenerativeModel(settings.gemini_model)
                print("Gemini client initialized for location conversion")
        except Exception as e:
            print(f"Gemini initialization error: {e}")
    
    async def initialize_session(self):
        """Initialize HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=settings.bus_data_timeout)
            )
    
    async def close_session(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def search_buses(self, start: str = None, end: str = None) -> Dict[str, Any]:
        """Search buses using your backend API."""
        try:
            await self.initialize_session()
            
            # Build URL
            url = f"{self.base_url}/api/buses/search"
            params = {}
            if start:
                params['start'] = start
            if end:
                params['end'] = end
            
            print(f"Searching buses: {url} with params: {params}")
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._process_bus_search_data(data)
                else:
                    print(f"API Error: HTTP {response.status}")
                    return self._get_fallback_data()
        
        except Exception as e:
            print(f"Bus search error: {e}")
            return self._get_fallback_data()
    
    async def get_bus_details(self, bus_id: str) -> Dict[str, Any]:
        """Get bus details by ID."""
        try:
            await self.initialize_session()
            
            url = f"{self.base_url}/api/buses/{bus_id}"
            print(f"Getting bus details: {url}")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._process_single_bus_data(data)
                else:
                    print(f"API Error: HTTP {response.status}")
                    return self._get_fallback_single_bus(bus_id)
        
        except Exception as e:
            print(f"Bus details error: {e}")
            return self._get_fallback_single_bus(bus_id)
    
    async def get_bus_by_name(self, name: str) -> Dict[str, Any]:
        """Get bus by name."""
        try:
            await self.initialize_session()
            
            url = f"{self.base_url}/api/buses/by-name/{name}"
            print(f"Getting bus by name: {url}")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return await self._process_single_bus_data(data)
                else:
                    print(f"API Error: HTTP {response.status}")
                    return self._get_fallback_single_bus(name)
        
        except Exception as e:
            print(f"Bus by name error: {e}")
            return self._get_fallback_single_bus(name)
    
    async def _process_bus_search_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process bus search results."""
        try:
            processed_buses = []
            
            # Handle different response formats
            buses_data = data.get('buses', data.get('data', []))
            if not isinstance(buses_data, list):
                buses_data = [buses_data] if buses_data else []
            
            for bus_data in buses_data:
                processed_bus = await self._convert_single_bus_data(bus_data)
                if processed_bus:
                    processed_buses.append(processed_bus)
            
            return {
                "success": True,
                "buses": processed_buses,
                "total_buses": len(processed_buses),
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "bus_ease_backend"
            }
            
        except Exception as e:
            print(f"Data processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "buses": [],
                "total_buses": 0
            }
    
    async def _process_single_bus_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process single bus data response."""
        try:
            # Handle different response formats
            bus_data = data.get('bus', data.get('data', data))
            
            processed_bus = await self._convert_single_bus_data(bus_data)
            
            if processed_bus:
                return {
                    "success": True,
                    "bus": processed_bus,
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "bus_ease_backend"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to process bus data",
                    "bus": None
                }
                
        except Exception as e:
            print(f"Single bus processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "bus": None
            }
    
    async def _convert_single_bus_data(self, bus_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert single bus data with Gemini location conversion."""
        try:
            if not bus_data:
                return None
            
            # Extract basic bus info
            bus_id = bus_data.get('id', bus_data.get('bus_id', 'unknown'))
            bus_name = bus_data.get('name', bus_data.get('bus_name', 'Unknown Bus'))
            bus_number = bus_data.get('number', bus_data.get('bus_number', bus_name))
            
            # Extract location data
            location_data = bus_data.get('location', bus_data.get('current_location', {}))
            
            # Handle different coordinate formats
            lat = None
            lng = None
            
            if isinstance(location_data, dict):
                lat = location_data.get('lat', location_data.get('latitude'))
                lng = location_data.get('lng', location_data.get('longitude'))
            elif isinstance(location_data, list) and len(location_data) >= 2:
                lat, lng = float(location_data[0]), float(location_data[1])
            
            # Try direct coordinates if location not found
            if not lat or not lng:
                lat = bus_data.get('lat', bus_data.get('latitude'))
                lng = bus_data.get('lng', bus_data.get('longitude'))
            
            if not lat or not lng:
                print(f"No valid coordinates found for bus: {bus_data}")
                lat, lng = 12.9716, 77.5946  # Default to Bangalore
            
            # Convert coordinates to location name using Gemini
            location_name = await self._coordinates_to_location_name(float(lat), float(lng))
            
            # Build processed bus data
            processed_bus = {
                "bus_id": str(bus_id),
                "bus_number": str(bus_number),
                "bus_name": str(bus_name),
                "status": bus_data.get('status', 'active'),
                "current_location": {
                    "lat": float(lat),
                    "lng": float(lng),
                    "location_name": location_name
                },
                "route": {
                    "start": bus_data.get('start', bus_data.get('source', 'Unknown')),
                    "end": bus_data.get('end', bus_data.get('destination', 'Unknown'))
                },
                "last_updated": bus_data.get('updated_at', time.strftime("%Y-%m-%d %H:%M:%S"))
            }
            
            return processed_bus
            
        except Exception as e:
            print(f"Bus conversion error: {e}")
            return None
    
    async def _coordinates_to_location_name(self, lat: float, lng: float) -> str:
        """Convert coordinates to location name using Gemini."""
        try:
            # Check cache first
            cache_key = f"{lat:.4f},{lng:.4f}"
            
            if cache_key in self.location_cache:
                cached_result = self.location_cache[cache_key]
                if time.time() - cached_result["timestamp"] < self.location_cache_duration:
                    return cached_result["location"]
            
            # Use Gemini for conversion
            if self.gemini_client and settings.is_gemini_location_enabled():
                try:
                    prompt = f"""
Convert these GPS coordinates to a specific location name in India:
Latitude: {lat}
Longitude: {lng}

Provide a detailed location name with landmarks (e.g., 'Central Railway Station, Bangalore').
Include the city name and keep it under 60 characters.
Focus on public transportation relevant locations.

Respond with ONLY the location name, nothing else.
"""
                    
                    response = self.gemini_client.generate_content(prompt)
                    
                    if response and response.text:
                        location_name = response.text.strip().strip('"').strip("'")
                        
                        # Validate and cache
                        if len(location_name) > 3 and len(location_name) < 60:
                            self.location_cache[cache_key] = {
                                "location": location_name,
                                "timestamp": time.time()
                            }
                            print(f"Gemini converted ({lat:.4f}, {lng:.4f}) -> {location_name}")
                            return location_name
                
                except Exception as e:
                    print(f"Gemini conversion error: {e}")
            
            # Fallback to basic location description
            return f"Location ({lat:.4f}, {lng:.4f}), India"
            
        except Exception as e:
            print(f"Location conversion error: {e}")
            return f"Unknown Location ({lat:.4f}, {lng:.4f})"
    
    def _get_fallback_data(self) -> Dict[str, Any]:
        """Generate fallback data when API is unavailable."""
        return {
            "success": True,
            "buses": [
                {
                    "bus_id": "DEMO001",
                    "bus_number": "101A",
                    "bus_name": "City Express",
                    "status": "active",
                    "current_location": {
                        "lat": 12.9716,
                        "lng": 77.5946,
                        "location_name": "Central Railway Station, Bangalore"
                    },
                    "route": {
                        "start": "City Center",
                        "end": "Airport"
                    },
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            ],
            "total_buses": 1,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "fallback_demo",
            "message": "Demo data - Backend API unavailable"
        }
    
    def _get_fallback_single_bus(self, bus_identifier: str) -> Dict[str, Any]:
        """Generate fallback data for single bus."""
        return {
            "success": True,
            "bus": {
                "bus_id": bus_identifier,
                "bus_number": bus_identifier,
                "bus_name": f"Bus {bus_identifier}",
                "status": "active",
                "current_location": {
                    "lat": 12.9716,
                    "lng": 77.5946,
                    "location_name": "Central Railway Station, Bangalore"
                },
                "route": {
                    "start": "City Center",
                    "end": "Airport"
                },
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "source": "fallback_demo"
        }

# Global instance
bus_data_fetcher = RealTimeBusDataFetcher()
