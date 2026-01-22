"""
BJ-SRI API fetcher

Qisqacha: BJ-SRI platformasidan API orqali tarixiy ma'lumotlarni olish uchun yordamchi klass.
Sozlashlar: BJ_SRI_BASE_URL va BJ_SRI_API_KEY environment o'zgaruvchilari orqali beriladi.
"""
import os
import requests
import json
from pathlib import Path
from datetime import datetime

class BJSRIFetcher:
    def __init__(self, base_url=None, api_key=None, output_dir=None):
        self.base_url = base_url or os.getenv('BJ_SRI_BASE_URL')
        self.api_key = api_key or os.getenv('BJ_SRI_API_KEY')
        self.output_dir = Path(output_dir or Path(__file__).resolve().parents[1] / 'raw_data')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.base_url:
            raise ValueError('BJ_SRI_BASE_URL not set')

    def _headers(self):
        headers = {'Accept': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

    def fetch(self, endpoint, params=None, save=True, filename=None):
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        resp = requests.get(url, headers=self._headers(), params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if save:
            fname = filename or f"bj_sri_{endpoint.replace('/','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            path = self.output_dir / fname
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return path
        return data

    def fetch_all(self, endpoints, common_params=None):
        """Fetch multiple endpoints and return list of saved file paths"""
        out = []
        for ep in endpoints:
            try:
                p = self.fetch(ep, params=common_params)
                out.append(p)
            except Exception as e:
                print(f"Failed to fetch {ep}: {e}")
        return out

# Example usage:
# fetcher = BJSRIFetcher()
# fetcher.fetch_all(['api/v1/rolls', 'api/v1/measurements'], common_params={'from':'2020-01-01','to':'2025-01-01'})
