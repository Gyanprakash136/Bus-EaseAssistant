"""
Bus Data Fetcher - Production Ready with Enhanced Error Handling
Integrated with bus-easebackend.onrender.com
"""
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
import json
import time
import logging
from datetime import datetime
from config.settings import settings

# Import Gemini with error handling
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class RealTimeBusDataFetcher:
    """Enhanced bus data fetcher with production-grade error handling."""
    
    def __init__(self):
        self.session = None
        self.base_url = settings.bus_api_base_url
        self.api_key = getattr(settings, 'bus_api_key', None)
        self.last_fetch_time = 0
        self.cache_duration = settings.bus_data_cache_duration
        self.cached_data = {}
        self.request_timeout = getattr(settings, 'bus_data_timeout', 10)
        
        # Initialize Gemini for location conversion
        self.gemini_client = None
        self.location_cache = {}
        self.location_cache_duration = settings.gemini_location_cache_duration
        self._initialize_gemini()
        
        # Stats tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "gemini_conversions": 0,
            "last_request_time": None
        }
        
        logger.info(f"✅ Bus Data Fetcher initialized")
        logger.info(f"   Backend URL: {self.base_url}")
        logger.info(f"   Cache Duration: {self.cache_duration}s")
        logger.info(f"   Request Timeout: {self.request_timeout}s")
    
    def _initialize_gemini(self):
        """Initialize Gemini for location conversion with error handling."""
        if not GENAI_AVAILABLE:
            logger.warning("⚠️ Gemini not available - using fallback location conversion")
            return
        
        try:
            if settings.is_gemini_location_enabled():
                genai.configure(api_key=settings.gemini_api_key)
                self.gemini_client = genai.GenerativeModel(settings.gemini_model)
                logger.info("✅ Gemini client initialized for location conversion")
            else:
                logger.warning("⚠️ Gemini API key not configured")
        except Exception as e:
            logger.error(f"❌ Gemini initialization error: {e}")
    
    async def initialize_session(self):
        """Initialize HTTP session with enhanced configuration."""
        if not self.session:
            # Enhanced session configuration
            connector = aiohttp.TCPConnector(
                limit=100,  # Connection pool limit
                limit_per_host=30,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
            )
            
            timeout = aiohttp.ClientTimeout(
                total=self.request_timeout,
                connect=5,  # Connection timeout
                sock_read=self.request_timeout  # Socket read timeout
            )
            
            headers = {
                'User-Agent': 'AI-Bus-Assistant/1.0',
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            # Add API key if available
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers
            )
            
            logger.info("✅ HTTP session initialized")
    
    async def close_session(self):
        """Close HTTP session gracefully."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("✅ HTTP session closed")
    
    async def search_buses(self, start: str = None, end: str = None) -> Dict[str, Any]:
        """Search buses with enhanced caching and error handling."""
        try:
            self.stats["total_requests"] += 1
            self.stats["last_request_time"] = datetime.now().isoformat()
            
            # Check cache first
            cache_key = f"search_{start}_{end}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.stats["cache_hits"] += 1
                logger.info(f"🔄 Cache hit for bus search: {start} → {end}")
                return cached_result
            
            await self.initialize_session()
            
            # Build URL and parameters
            url = f"{self.base_url}/api/buses/search"
            params = {}
            if start:
                params['start'] = start.strip()
            if end:
                params['end'] = end.strip()
            
            logger.info(f"🔍 Searching buses: {url} with params: {params}")
            
            async with self.session.get(url, params=params) as response:
                response_text = await response.text()
                
                if response.status == 200:
                    try:
                        data = await response.json()
                        result = await self._process_bus_search_data(data)
                        
                        # Cache successful result
                        self._cache_result(cache_key, result)
                        self.stats["successful_requests"] += 1
                        
                        return result
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Invalid JSON response: {e}")
                        raise
                        
                elif response.status == 404:
                    logger.warning(f"⚠️ No buses found: {response.status}")
                    result = {
                        "success": True,
                        "buses": [],
                        "total_buses": 0,
                        "message": "No buses found for the specified route",
                        "source": "bus_ease_backend"
                    }
                    self._cache_result(cache_key, result)
                    self.stats["successful_requests"] += 1
                    return result
                    
                else:
                    logger.error(f"❌ API Error: HTTP {response.status} - {response_text[:200]}")
                    self.stats["failed_requests"] += 1
                    return self._get_fallback_data(f"API returned HTTP {response.status}")
        
        except asyncio.TimeoutError:
            logger.error(f"❌ Bus search timeout after {self.request_timeout}s")
            self.stats["failed_requests"] += 1
            return self._get_fallback_data("Request timeout")
            
        except aiohttp.ClientError as e:
            logger.error(f"❌ Bus search client error: {e}")
            self.stats["failed_requests"] += 1
            return self._get_fallback_data(f"Client error: {str(e)}")
            
        except Exception as e:
            logger.error(f"❌ Bus search unexpected error: {e}")
            self.stats["failed_requests"] += 1
            return self._get_fallback_data(f"Unexpected error: {str(e)}")
    
    async def get_bus_details(self, bus_id: str) -> Dict[str, Any]:
        """Get bus details by ID with enhanced error handling."""
        try:
            self.stats["total_requests"] += 1
            self.stats["last_request_time"] = datetime.now().isoformat()
            
            # Check cache
            cache_key = f"details_{bus_id}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.stats["cache_hits"] += 1
                logger.info(f"🔄 Cache hit for bus details: {bus_id}")
                return cached_result
            
            await self.initialize_session()
            
            url = f"{self.base_url}/api/buses/{bus_id}"
            logger.info(f"🔍 Getting bus details: {url}")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    result = await self._process_single_bus_data(data)
                    
                    # Cache result
                    self._cache_result(cache_key, result)
                    self.stats["successful_requests"] += 1
                    
                    return result
                    
                elif response.status == 404:
                    logger.warning(f"⚠️ Bus not found: {bus_id}")
                    result = {
                        "success": False,
                        "error": f"Bus {bus_id} not found",
                        "bus": None
                    }
                    self.stats["successful_requests"] += 1
                    return result
                    
                else:
                    logger.error(f"❌ API Error: HTTP {response.status}")
                    self.stats["failed_requests"] += 1
                    return self._get_fallback_single_bus(bus_id, f"API error: HTTP {response.status}")
        
        except Exception as e:
            logger.error(f"❌ Bus details error: {e}")
            self.stats["failed_requests"] += 1
            return self._get_fallback_single_bus(bus_id, str(e))
    
    async def get_bus_by_name(self, name: str) -> Dict[str, Any]:
        """Get bus by name with enhanced error handling."""
        try:
            self.stats["total_requests"] += 1
            self.stats["last_request_time"] = datetime.now().isoformat()
            
            # Clean and validate name
            clean_name = name.strip().upper() if name else ""
            if not clean_name:
                return {
                    "success": False,
                    "error": "Bus name cannot be empty",
                    "bus": None
                }
            
            # Check cache
            cache_key = f"name_{clean_name}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.stats["cache_hits"] += 1
                logger.info(f"🔄 Cache hit for bus name: {clean_name}")
                return cached_result
            
            await self.initialize_session()
            
            url = f"{self.base_url}/api/buses/by-name/{clean_name}"
            logger.info(f"🔍 Getting bus by name: {url}")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    result = await self._process_single_bus_data(data)
                    
                    # Cache result
                    self._cache_result(cache_key, result)
                    self.stats["successful_requests"] += 1
                    
                    return result
                    
                elif response.status == 404:
                    logger.warning(f"⚠️ Bus not found: {clean_name}")
                    result = {
                        "success": False,
                        "error": f"Bus '{clean_name}' not found",
                        "bus": None,
                        "suggestion": "Please check the bus number/name and try again"
                    }
                    self.stats["successful_requests"] += 1
                    return result
                    
                else:
                    logger.error(f"❌ API Error: HTTP {response.status}")
                    self.stats["failed_requests"] += 1
                    return self._get_fallback_single_bus(clean_name, f"API error: HTTP {response.status}")
        
        except Exception as e:
            logger.error(f"❌ Bus by name error: {e}")
            self.stats["failed_requests"] += 1
            return self._get_fallback_single_bus(name, str(e))
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if still valid."""
        if cache_key in self.cached_data:
            cached_item = self.cached_data[cache_key]
            if time.time() - cached_item["timestamp"] < self.cache_duration:
                return cached_item["data"]
            else:
                # Remove expired cache
                del self.cached_data[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, data: Dict[str, Any]):
        """Cache result with timestamp."""
        self.cached_data[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }
        
        # Limit cache size
        if len(self.cached_data) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self.cached_data.keys(), 
                key=lambda k: self.cached_data[k]["timestamp"]
            )[:10]
            for key in oldest_keys:
                del self.cached_data[key]
    
    async def _process_bus_search_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process bus search results with enhanced error handling."""
        try:
            processed_buses = []
            
            # Handle different response formats
            buses_data = data.get('buses', data.get('data', data.get('results', [])))
            
            # Handle single bus response
            if isinstance(buses_data, dict):
                buses_data = [buses_data]
            elif not isinstance(buses_data, list):
                buses_data = []
            
            logger.info(f"📊 Processing {len(buses_data)} buses from API response")
            
            for bus_data in buses_data:
                if bus_data:  # Skip None/empty entries
                    processed_bus = await self._convert_single_bus_data(bus_data)
                    if processed_bus:
                        processed_buses.append(processed_bus)
            
            return {
                "success": True,
                "buses": processed_buses,
                "total_buses": len(processed_buses),
                "last_updated": datetime.now().isoformat(),
                "source": "bus_ease_backend",
                "cache_duration": self.cache_duration
            }
            
        except Exception as e:
            logger.error(f"❌ Data processing error: {e}")
            return {
                "success": False,
                "error": f"Data processing failed: {str(e)}",
                "buses": [],
                "total_buses": 0
            }
    
    async def _process_single_bus_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process single bus data response with enhanced handling."""
        try:
            # Handle different response formats
            bus_data = data.get('bus', data.get('data', data))
            
            if not bus_data:
                return {
                    "success": False,
                    "error": "No bus data in response",
                    "bus": None
                }
            
            processed_bus = await self._convert_single_bus_data(bus_data)
            
            if processed_bus:
                return {
                    "success": True,
                    "bus": processed_bus,
                    "last_updated": datetime.now().isoformat(),
                    "source": "bus_ease_backend"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to process bus data",
                    "bus": None
                }
                
        except Exception as e:
            logger.error(f"❌ Single bus processing error: {e}")
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}",
                "bus": None
            }
    
    async def _convert_single_bus_data(self, bus_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert single bus data with enhanced Gemini location conversion."""
        try:
            if not bus_data:
                return None
            
            # Extract basic bus info with multiple fallbacks
            bus_id = str(bus_data.get('id', bus_data.get('bus_id', bus_data.get('_id', 'unknown'))))
            bus_name = str(bus_data.get('name', bus_data.get('bus_name', bus_data.get('busName', 'Unknown Bus'))))
            bus_number = str(bus_data.get('number', bus_data.get('bus_number', bus_data.get('busNumber', bus_name))))
            
            # Extract location data with multiple format support
            lat, lng = self._extract_coordinates(bus_data)
            
            # Convert coordinates to location name using Gemini
            location_name = await self._coordinates_to_location_name(lat, lng)
            
            # Extract route information
            route_info = self._extract_route_info(bus_data)
            
            # Build processed bus data
            processed_bus = {
                "bus_id": bus_id,
                "bus_number": bus_number,
                "bus_name": bus_name,
                "status": bus_data.get('status', bus_data.get('busStatus', 'active')),
                "current_location": {
                    "lat": lat,
                    "lng": lng,
                    "location_name": location_name,
                    "coordinates": f"{lat:.6f}, {lng:.6f}"
                },
                "route": route_info,
                "last_updated": bus_data.get('updated_at', bus_data.get('lastUpdated', datetime.now().isoformat())),
                "additional_info": {
                    "capacity": bus_data.get('capacity'),
                    "driver": bus_data.get('driver'),
                    "contact": bus_data.get('contact')
                }
            }
            
            logger.info(f"✅ Processed bus: {bus_number} at {location_name}")
            return processed_bus
            
        except Exception as e:
            logger.error(f"❌ Bus conversion error: {e}")
            return None
    
    def _extract_coordinates(self, bus_data: Dict[str, Any]) -> tuple[float, float]:
        """Extract coordinates from various possible formats."""
        lat = lng = None
        
        # Try location object
        location_data = bus_data.get('location', bus_data.get('current_location', bus_data.get('position', {})))
        
        if isinstance(location_data, dict):
            lat = location_data.get('lat', location_data.get('latitude'))
            lng = location_data.get('lng', location_data.get('longitude', location_data.get('long')))
        elif isinstance(location_data, list) and len(location_data) >= 2:
            try:
                lat, lng = float(location_data[0]), float(location_data[1])
            except (ValueError, TypeError):
                pass
        
        # Try direct coordinates
        if not lat or not lng:
            lat = bus_data.get('lat', bus_data.get('latitude'))
            lng = bus_data.get('lng', bus_data.get('longitude', bus_data.get('long')))
        
        # Try GPS coordinates
        if not lat or not lng:
            gps = bus_data.get('gps', bus_data.get('coordinates', {}))
            if isinstance(gps, dict):
                lat = gps.get('lat', gps.get('latitude'))
                lng = gps.get('lng', gps.get('longitude'))
        
        # Convert to float and validate
        try:
            lat = float(lat) if lat is not None else None
            lng = float(lng) if lng is not None else None
        except (ValueError, TypeError):
            lat = lng = None
        
        # Default coordinates if none found (Bangalore city center)
        if not lat or not lng or lat == 0 or lng == 0:
            logger.warning(f"⚠️ No valid coordinates found, using default: {bus_data}")
            lat, lng = 12.9716, 77.5946
        
        return float(lat), float(lng)
    
    def _extract_route_info(self, bus_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract route information from bus data."""
        route = bus_data.get('route', {})
        
        if isinstance(route, dict):
            start = route.get('start', route.get('source', route.get('origin', 'Unknown')))
            end = route.get('end', route.get('destination', route.get('dest', 'Unknown')))
        else:
            # Try direct fields
            start = bus_data.get('start', bus_data.get('source', bus_data.get('origin', 'Unknown')))
            end = bus_data.get('end', bus_data.get('destination', bus_data.get('dest', 'Unknown')))
        
        return {
            "start": str(start),
            "end": str(end),
            "route_id": str(bus_data.get('route_id', bus_data.get('routeId', 'unknown')))
        }
    
    async def _coordinates_to_location_name(self, lat: float, lng: float) -> str:
        """Convert coordinates to location name using Gemini with enhanced caching."""
        try:
            # Check cache first
            cache_key = f"{lat:.4f},{lng:.4f}"
            
            if cache_key in self.location_cache:
                cached_result = self.location_cache[cache_key]
                if time.time() - cached_result["timestamp"] < self.location_cache_duration:
                    return cached_result["location"]
                else:
                    # Remove expired cache
                    del self.location_cache[cache_key]
            
            # Use Gemini for conversion
            if self.gemini_client and settings.is_gemini_location_enabled():
                try:
                    prompt = settings.get_location_prompt_template() + f"""
GPS Coordinates: {lat}, {lng}
Location Context: India, Urban area
Expected Format: Landmark/Area Name, City Name
"""
                    
                    response = self.gemini_client.generate_content(prompt)
                    
                    if response and response.text:
                        location_name = response.text.strip().strip('"').strip("'")
                        
                        # Validate location name
                        if 3 <= len(location_name) <= 60 and location_name.lower() not in ['unknown', 'error', 'none']:
                            # Cache result
                            self.location_cache[cache_key] = {
                                "location": location_name,
                                "timestamp": time.time()
                            }
                            
                            self.stats["gemini_conversions"] += 1
                            logger.info(f"🌍 Gemini location: ({lat:.4f}, {lng:.4f}) → {location_name}")
                            return location_name
                
                except Exception as e:
                    logger.warning(f"⚠️ Gemini conversion error: {e}")
            
            # Fallback to basic location description
            fallback_location = f"Location ({lat:.4f}, {lng:.4f}), India"
            logger.info(f"📍 Fallback location: {fallback_location}")
            return fallback_location
            
        except Exception as e:
            logger.error(f"❌ Location conversion error: {e}")
            return f"Unknown Location ({lat:.4f}, {lng:.4f})"
    
    def _get_fallback_data(self, reason: str = "Backend unavailable") -> Dict[str, Any]:
        """Generate enhanced fallback data when API is unavailable."""
        return {
            "success": True,
            "buses": [
                {
                    "bus_id": "DEMO001",
                    "bus_number": "101A",
                    "bus_name": "City Express Demo",
                    "status": "active",
                    "current_location": {
                        "lat": 12.9716,
                        "lng": 77.5946,
                        "location_name": "Central Railway Station, Bangalore",
                        "coordinates": "12.971600, 77.594600"
                    },
                    "route": {
                        "start": "City Center",
                        "end": "Airport",
                        "route_id": "demo_route"
                    },
                    "last_updated": datetime.now().isoformat(),
                    "additional_info": {
                        "capacity": 50,
                        "driver": "Demo Driver",
                        "contact": "N/A"
                    }
                },
                {
                    "bus_id": "DEMO002", 
                    "bus_number": "202B",
                    "bus_name": "Metro Connector",
                    "status": "active",
                    "current_location": {
                        "lat": 12.9352,
                        "lng": 77.6245,
                        "location_name": "Electronic City, Bangalore",
                        "coordinates": "12.935200, 77.624500"
                    },
                    "route": {
                        "start": "Silk Board",
                        "end": "Electronic City",
                        "route_id": "demo_route_2"
                    },
                    "last_updated": datetime.now().isoformat(),
                    "additional_info": {
                        "capacity": 40,
                        "driver": "Demo Driver 2",
                        "contact": "N/A"
                    }
                }
            ],
            "total_buses": 2,
            "last_updated": datetime.now().isoformat(),
            "source": "fallback_demo",
            "message": f"Demo data - {reason}",
            "note": "This is sample data for testing. Real data will be available when the backend API is accessible."
        }
    
    def _get_fallback_single_bus(self, bus_identifier: str, reason: str = "Backend unavailable") -> Dict[str, Any]:
        """Generate enhanced fallback data for single bus."""
        return {
            "success": True,
            "bus": {
                "bus_id": bus_identifier,
                "bus_number": bus_identifier,
                "bus_name": f"Bus {bus_identifier} (Demo)",
                "status": "active",
                "current_location": {
                    "lat": 12.9716,
                    "lng": 77.5946,
                    "location_name": "Central Railway Station, Bangalore",
                    "coordinates": "12.971600, 77.594600"
                },
                "route": {
                    "start": "City Center",
                    "end": "Airport",
                    "route_id": f"route_{bus_identifier}"
                },
                "last_updated": datetime.now().isoformat(),
                "additional_info": {
                    "capacity": 45,
                    "driver": "Demo Driver",
                    "contact": "N/A"
                }
            },
            "source": "fallback_demo",
            "message": f"Demo data for bus {bus_identifier} - {reason}",
            "note": "This is sample data for testing purposes."
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fetcher statistics."""
        return {
            **self.stats,
            "cache_size": len(self.cached_data),
            "location_cache_size": len(self.location_cache),
            "success_rate": (
                self.stats["successful_requests"] / max(self.stats["total_requests"], 1) * 100
                if self.stats["total_requests"] > 0 else 0
            ),
            "configuration": {
                "base_url": self.base_url,
                "cache_duration": self.cache_duration,
                "request_timeout": self.request_timeout,
                "gemini_enabled": self.gemini_client is not None
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the bus data service."""
        try:
            await self.initialize_session()
            
            # Test connection with a simple request
            start_time = time.time()
            async with self.session.get(f"{self.base_url}/api/buses/search", params={"limit": 1}) as response:
                response_time = time.time() - start_time
                
                return {
                    "status": "healthy" if response.status < 500 else "degraded",
                    "response_code": response.status,
                    "response_time_ms": round(response_time * 1000, 2),
                    "backend_url": self.base_url,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "backend_url": self.base_url,
                "timestamp": datetime.now().isoformat()
            }

# Global instance
bus_data_fetcher = RealTimeBusDataFetcher()
