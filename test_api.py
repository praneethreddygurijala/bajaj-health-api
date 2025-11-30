import requests
import json
import sys

# URL of your local API
url = "http://localhost:8000/extract-bill-data"

# Default Sample URL (Official)
sample_doc = "https://www.wmaccess.com/downloads/sample-invoice.pdf"

# Accept URL from command line if provided
if len(sys.argv) > 1:
    sample_doc = sys.argv[1]

print(f"🚀 Testing API with document: {sample_doc[:50]}...")

try:
    response = requests.post(url, json={"document": sample_doc})
    
    if response.status_code == 200:
        data = response.json()
        if data.get("is_success"):
            print("\n✅ SUCCESS! (200 OK)")
            
            # Print Token Usage (New Requirement)
            tokens = data.get("token_usage", {})
            print(f"📊 Token Usage: Input={tokens.get('input_tokens')}, Output={tokens.get('output_tokens')}, Total={tokens.get('total_tokens')}")
            
            # Print Data
            result = data.get("data", {})
            print(f"🧾 Total Items: {result.get('total_item_count')}")
            
            for page in result.get("pagewise_line_items", []):
                print(f"\n[Page {page.get('page_no')} - Type: {page.get('page_type')}]")
                for item in page.get("bill_items", []):
                    print(f" - {item['item_name']:<30} | ₹{item['item_amount']}")
        else:
            print("❌ Success is False:", data)
    else:
        print(f"❌ HTTP Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"💥 Error: {e}")