#!/usr/bin/env python3
"""
Simple test script for the MEGAN inference API.
"""

import requests
import time
import json
from typing import List, Dict

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_V1_URL = f"{API_BASE_URL}/api/v1"

# Test molecules (SMILES)
TEST_MOLECULES = [
    "CCO",  # Ethanol
    "CC(=O)O",  # Acetic acid
    "c1ccccc1",  # Benzene
    "CCN(CC)CC",  # Triethylamine
    "O",  # Water
    "CCCCCCCC",  # Octane
    "CC(C)C",  # Isobutane
]

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Health check passed: {data['status']}")
        print(f"   Model loaded: {data['model_loaded']}")
        print(f"   Model version: {data.get('model_version', 'N/A')}")
        print(f"   Uptime: {data['uptime_seconds']:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_model_info():
    """Test the model info endpoint."""
    print("\n🔍 Testing model info endpoint...")
    try:
        response = requests.get(f"{API_V1_URL}/model/info")
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Model info retrieved:")
        print(f"   Model loaded: {data['model_loaded']}")
        print(f"   Model version: {data['model_version']}")
        print(f"   Model type: {data['model_type']}")
        return True
        
    except Exception as e:
        print(f"❌ Model info test failed: {e}")
        return False

def test_single_prediction(smiles: str):
    """Test prediction for a single molecule."""
    print(f"\n🧪 Testing prediction for: {smiles}")
    try:
        payload = {"smiles": smiles}
        response = requests.post(f"{API_V1_URL}/predict", json=payload)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Prediction successful:")
        print(f"   SMILES: {data['smiles']}")
        print(f"   Solubility: {data['prediction']['value']:.3f} {data['prediction']['unit']}")
        print(f"   Confidence: {data['prediction']['confidence']:.3f}")
        print(f"   Processing time: {data['processing_time_ms']:.2f}ms")
        print(f"   Model version: {data['model_version']}")
        
        return data
        
    except Exception as e:
        print(f"❌ Prediction failed for {smiles}: {e}")
        return None

def test_invalid_smiles():
    """Test prediction with invalid SMILES."""
    print(f"\n🚫 Testing invalid SMILES...")
    invalid_smiles = ["INVALID", "", "XYZ123"]
    
    for smiles in invalid_smiles:
        try:
            payload = {"smiles": smiles}
            response = requests.post(f"{API_V1_URL}/predict", json=payload)
            
            if response.status_code == 422:
                print(f"✅ Correctly rejected invalid SMILES: '{smiles}'")
            else:
                print(f"⚠️  Unexpected response for '{smiles}': {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error testing invalid SMILES '{smiles}': {e}")

def run_performance_test(molecules: List[str], num_iterations: int = 3):
    """Run a simple performance test."""
    print(f"\n⚡ Running performance test ({num_iterations} iterations)...")
    
    total_time = 0
    successful_predictions = 0
    
    for iteration in range(num_iterations):
        print(f"   Iteration {iteration + 1}/{num_iterations}")
        
        for smiles in molecules:
            start_time = time.time()
            result = test_single_prediction(smiles)
            end_time = time.time()
            
            if result:
                successful_predictions += 1
                total_time += (end_time - start_time)
    
    if successful_predictions > 0:
        avg_time = total_time / successful_predictions
        print(f"\n📊 Performance Results:")
        print(f"   Total predictions: {successful_predictions}")
        print(f"   Average time per prediction: {avg_time:.3f}s")
        print(f"   Predictions per second: {1/avg_time:.2f}")

def main():
    """Run all tests."""
    print("🚀 Starting MEGAN Inference API Tests")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health_check():
        print("❌ Health check failed. Is the server running?")
        return
    
    # Test 2: Model info
    test_model_info()
    
    # Test 3: Valid predictions
    print(f"\n🧪 Testing predictions for {len(TEST_MOLECULES)} molecules...")
    results = []
    for smiles in TEST_MOLECULES:
        result = test_single_prediction(smiles)
        if result:
            results.append(result)
    
    # Test 4: Invalid SMILES
    test_invalid_smiles()
    
    # Test 5: Performance test
    if results:
        run_performance_test(TEST_MOLECULES[:3], num_iterations=2)
    
    print("\n🎉 Testing completed!")
    print(f"   Successful predictions: {len(results)}")

if __name__ == "__main__":
    main()
