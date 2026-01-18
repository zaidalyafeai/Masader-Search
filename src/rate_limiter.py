import streamlit as st  
from typing import Dict
import time

class RateLimiter:
    def __init__(self, requests: int, time_window: int = 60):
        """
        Initialize rate limiter
        :param requests: Number of requests allowed per time window
        :param time_window: Time window in seconds (default: 60 seconds)
        """
        self.requests = requests
        self.time_window = time_window
        self.requests_made: Dict[str, list] = {}

    def is_allowed(self, key: str) -> bool:
        """
        Check if the request is allowed based on rate limit
        :param key: Unique identifier for the client (can be IP, user ID, etc.)
        :return: True if request is allowed, False otherwise
        """
        current_time = time.time()
        
        # Remove timestamps older than the time window
        if key in self.requests_made:
            self.requests_made[key] = [
                t for t in self.requests_made[key] 
                if current_time - t < self.time_window
            ]
        else:
            self.requests_made[key] = []
        
        # Check if we've exceeded the rate limit
        if len(self.requests_made[key]) >= self.requests:
            return False
        
        # Add the current request timestamp
        self.requests_made[key].append(current_time)
        return True

    def get_retry_after(self, key: str) -> int:
        """
        Get the number of seconds until the next request is allowed
        :param key: Unique identifier for the client
        :return: Number of seconds until next allowed request
        """
        if key not in self.requests_made or not self.requests_made[key]:
            return 0
            
        current_time = time.time()
        oldest_request = min(self.requests_made[key])
        return int((oldest_request + self.time_window) - current_time)

def get_client_ip() -> str:
    """
    Get the client's IP address from Streamlit's session state
    Note: In a production environment, you might want to get the real IP from the request headers
    """
    if '_client_ip' not in st.session_state:
        st.session_state._client_ip = str(id(st.session_state))
    return st.session_state._client_ip
