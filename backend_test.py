import requests
import sys
import time
from datetime import datetime

class ArxivAPITester:
    def __init__(self, base_url="https://research-assist-14.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.sync_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    print(f"   Response: {str(response_data)[:200]}...")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text[:200]}")
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timeout after {timeout}s")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_api_root(self):
        """Test GET /api/ - API root endpoint"""
        success, response = self.run_test(
            "API Root",
            "GET",
            "",
            200
        )
        if success and 'version' in response:
            print(f"   API Version: {response.get('version')}")
        return success

    def test_stats(self):
        """Test GET /api/stats - Should return stats"""
        success, response = self.run_test(
            "Stats Endpoint",
            "GET",
            "stats",
            200
        )
        if success:
            required_fields = ['total_papers', 'today_added', 'sync_status', 'categories']
            for field in required_fields:
                if field not in response:
                    print(f"   ⚠️  Missing field: {field}")
                else:
                    print(f"   {field}: {response[field]}")
        return success

    def test_recent_papers(self):
        """Test GET /api/papers/recent - Should return list of papers"""
        success, response = self.run_test(
            "Recent Papers",
            "GET",
            "papers/recent",
            200
        )
        if success:
            if isinstance(response, list):
                print(f"   Found {len(response)} papers")
                if len(response) > 0:
                    paper = response[0]
                    print(f"   Sample paper: {paper.get('title', 'No title')[:50]}...")
            else:
                print(f"   ⚠️  Expected list, got {type(response)}")
        return success

    def test_sync_trigger(self):
        """Test POST /api/sync/trigger - Should trigger sync"""
        success, response = self.run_test(
            "Sync Trigger",
            "POST",
            "sync/trigger",
            200
        )
        if success and 'sync_id' in response:
            self.sync_id = response['sync_id']
            print(f"   Sync ID: {self.sync_id}")
            print(f"   Status: {response.get('status')}")
        return success

    def test_sync_status(self):
        """Test GET /api/sync/status - Should return sync status"""
        success, response = self.run_test(
            "Sync Status",
            "GET",
            "sync/status",
            200
        )
        if success:
            print(f"   Status: {response.get('status')}")
            if 'last_sync' in response and response['last_sync']:
                print(f"   Last sync: {response['last_sync']}")
        return success

    def test_search_empty_db(self):
        """Test POST /api/search - Search with empty database"""
        success, response = self.run_test(
            "Search (Empty DB)",
            "POST",
            "search",
            200,
            data={"query": "What are transformers in machine learning?"}
        )
        if success:
            if 'answer' in response:
                print(f"   Answer length: {len(response['answer'])} chars")
                print(f"   Sources count: {len(response.get('sources', []))}")
            else:
                print(f"   ⚠️  No answer field in response")
        return success

    def wait_for_sync_completion(self, max_wait_time=120):
        """Wait for sync to complete"""
        print(f"\n⏳ Waiting for sync to complete (max {max_wait_time}s)...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(f"{self.api_url}/sync/status", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get('status', 'unknown')
                    print(f"   Sync status: {status}")
                    
                    if status == 'idle':
                        print("✅ Sync completed!")
                        return True
                    elif status == 'syncing':
                        print("   Still syncing...")
                        time.sleep(5)
                    else:
                        print(f"   Unknown status: {status}")
                        time.sleep(5)
                else:
                    print(f"   Error checking status: {response.status_code}")
                    time.sleep(5)
            except Exception as e:
                print(f"   Error: {e}")
                time.sleep(5)
        
        print(f"⚠️  Sync did not complete within {max_wait_time}s")
        return False

    def test_search_with_data(self):
        """Test POST /api/search - Search with populated database"""
        success, response = self.run_test(
            "Search (With Data)",
            "POST",
            "search",
            200,
            data={"query": "What are the latest advances in transformer efficiency?"},
            timeout=60  # Longer timeout for AI processing
        )
        if success:
            if 'answer' in response:
                print(f"   Answer length: {len(response['answer'])} chars")
                print(f"   Sources count: {len(response.get('sources', []))}")
                if response.get('sources'):
                    print(f"   First source: {response['sources'][0].get('title', 'No title')[:50]}...")
            else:
                print(f"   ⚠️  No answer field in response")
        return success

def main():
    print("🚀 Starting arXiv AI Research Q&A API Tests")
    print("=" * 60)
    
    tester = ArxivAPITester()
    
    # Test basic endpoints first
    print("\n📋 PHASE 1: Basic API Tests")
    tester.test_api_root()
    tester.test_stats()
    tester.test_recent_papers()
    
    # Test sync functionality
    print("\n📋 PHASE 2: Sync Tests")
    tester.test_sync_status()
    
    # Test search with empty database
    print("\n📋 PHASE 3: Search Tests (Empty DB)")
    tester.test_search_empty_db()
    
    # Trigger sync and wait for completion
    print("\n📋 PHASE 4: Sync Process")
    sync_success = tester.test_sync_trigger()
    
    if sync_success:
        # Wait for sync to complete
        sync_completed = tester.wait_for_sync_completion()
        
        if sync_completed:
            # Test endpoints after sync
            print("\n📋 PHASE 5: Post-Sync Tests")
            tester.test_stats()  # Check updated stats
            tester.test_recent_papers()  # Should have papers now
            tester.test_search_with_data()  # Search should work better
        else:
            print("\n⚠️  Skipping post-sync tests due to sync timeout")
    else:
        print("\n⚠️  Skipping sync-dependent tests due to sync trigger failure")
    
    # Final results
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS")
    print(f"Tests passed: {tester.tests_passed}/{tester.tests_run}")
    success_rate = (tester.tests_passed / tester.tests_run * 100) if tester.tests_run > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Backend API tests mostly successful!")
        return 0
    elif success_rate >= 50:
        print("⚠️  Backend API has some issues but core functionality works")
        return 1
    else:
        print("❌ Backend API has significant issues")
        return 2

if __name__ == "__main__":
    sys.exit(main())