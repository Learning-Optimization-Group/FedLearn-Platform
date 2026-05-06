#!/usr/bin/env python3
"""
FedLearn Full Platform E2E Test Suite
=====================================
Tests the full platform integration:
  - Registers/Logs in a test user via the Spring Boot Backend (default :8081)
  - Creates a new Training Project via the REST API
  - Starts the Flower gRPC Server natively mapped through the Java FlowerServerManager
  - Spawns local simulated Python clients (e.g. SimpleCNN models) to train 
  - Polls the backend API for RoundResults to verify successful persistence
  - Automatically handles cleanup

Usage:
    cd framework/
    python run_platform_e2e_test.py
"""

import os
import sys
import time
import uuid
import json
import random
import string
import requests
import multiprocessing
import traceback

# Add framework to path for clients
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8081/api")

# Quick random string generator for unique tests
def random_string(n=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

# -----------------
# CLIENT SIMULATOR
# -----------------
def spawn_mnist_client(client_idx, client_id, server_port):
    """Spawns a local python client mimicking an edge device."""
    try:
        from fedlearn.client.client import start_client
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, Subset
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim

        # We must mimic the CnnNet structure from the backend init_model.py
        class CnnNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 16 * 5 * 5)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)

        class MnistClient:
            def __init__(self, cid):
                self.net = CnnNet()
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                # Using CIFAR10 to match backend CNN expectations
                full = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
                sz = len(full) // 2  # 2 clients
                self.loader = DataLoader(Subset(full, range(cid * sz, (cid + 1) * sz)), batch_size=32, shuffle=True)

            def get_parameters(self): return self.net.state_dict()
            
            def fit(self, params, config):
                self.net.load_state_dict(params)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(self.net.parameters(), lr=0.01)
                self.net.train()
                for _ in range(1): # Just 1 epoch locally for speed
                    for imgs, lbs in self.loader:
                        optimizer.zero_grad()
                        loss = criterion(self.net(imgs), lbs)
                        loss.backward()
                        optimizer.step()
                return self.net.state_dict(), len(self.loader.dataset)

        print(f"[Client {client_id}] Connecting to localhost:{server_port}...")
        time.sleep(2) # Buffer connection
        start_client(f"localhost:{server_port}", MnistClient(client_idx), client_id)
        print(f"[Client {client_id}] Finished successfully.")
    except Exception as e:
        print(f"[Client {client_id}] Error: {e}")
        traceback.print_exc()

def e2e_test_flow():
    print("===========================================")
    print("  FEDLEARN PLATFORM E2E INTEGRATION TEST")
    print("===========================================\n")
    
    session = requests.Session()
    
    # 1. Register & Login
    username = f"e2e_user_{random_string()}"
    password = "e2e_Password123!"
    email = f"{username}@test.com"
    
    print(f"[*] Registering user {username} to {API_BASE_URL}...")
    reg_res = session.post(f"{API_BASE_URL}/auth/register", json={
        "username": username,
        "email": email,
        "password": password
    })
    
    if reg_res.status_code not in (200, 201):
        print(f"[!] Registration failed: {reg_res.text}")
        return False
        
    print(f"[*] Logging in...")
    log_res = session.post(f"{API_BASE_URL}/auth/login", json={
        "username": username,
        "password": password
    })
    
    if log_res.status_code != 200:
        print(f"[!] Login failed: {log_res.text}")
        return False
        
    token = log_res.json().get("accessToken")
    session.headers.update({"Authorization": f"Bearer {token}"})
    print("    [+] Authentication successful.")
    
    # 2. Create a Project
    print("[*] Creating a new Federated Learning Project via API...")
    proj_res = session.post(f"{API_BASE_URL}/projects", json={
        "name": f"E2E Integration Test {random_string()}",
        "modelType": "CNN",
        "modelName": "SimpleCNN",
        "optimizer": "SGD",
        "pretrainEpochs": 0
    })
    
    if proj_res.status_code != 200:
        print(f"[!] Project creation failed: {proj_res.text}")
        return False
        
    project_id = proj_res.json().get("id")
    print(f"    [+] Project created! ID: {project_id}")
    
    # 3. Start the Server (Flower wrapper on Java backend)
    print(f"[*] Starting the Flower Server process for Project {project_id}...")
    num_rounds = 2
    start_res = session.post(f"{API_BASE_URL}/projects/{project_id}/start", json={
        "strategy": "FedAvg",
        "numRounds": num_rounds,
        "minClients": 2
    })
    
    if start_res.status_code != 200:
        print(f"[!] Failed to start project server: {start_res.text}")
        return False
        
    server_port = start_res.json().get("serverPort")
    print(f"    [+] Backend successfully spawned Python gRPC server on Port {server_port}!")
    
    # 4. Spawning Edge Clients
    print(f"[*] Spawning 2 simulated Python edge clients...")
    
    # Pre-download CIFAR10 dataset from the main process to prevent multiprocess race conditions
    print("    [+] Pre-downloading CIFAR10 dataset if missing...")
    from torchvision import datasets
    datasets.CIFAR10(root="./data", train=True, download=True)

    procs = []
    for i in range(2):
        p = multiprocessing.Process(target=spawn_mnist_client, args=(i, f"edge_device_{i}", server_port))
        p.start()
        procs.append(p)
    
    # 5. Monitor Results via API
    print(f"[*] Waiting for training to complete and verifying persistent database metrics...")
    
    max_retries = 60
    rounds_completed = 0
    test_passed = False
    
    try:
        for attempt in range(max_retries):
            time.sleep(5)
            # Poll results
            res = session.get(f"{API_BASE_URL}/projects/{project_id}/results")
            if res.status_code == 200:
                results_array = res.json()
                rounds_completed = len(results_array)
                print(f"    ... Polling ({attempt+1}/{max_retries}): {rounds_completed}/{num_rounds} rounds stored in DB.")
                if rounds_completed >= num_rounds:
                    print(f"    [+] Training fully completed! Server synced results properly.")
                    test_passed = True
                    break
    except KeyboardInterrupt:
        print("[!] Interrupted.")
    finally:
        # 6. Cleanup
        print(f"[*] Stopping Project Server...")
        session.post(f"{API_BASE_URL}/projects/{project_id}/stop")
        
        print(f"[*] Force-killing edge client orphans if any...")
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join()
                
        print(f"[*] Attempting to delete test project {project_id}...")
        session.post(f"{API_BASE_URL}/projects/{project_id}/delete")
        
    
    print("\n===========================================")
    if test_passed:
        print("         🎉 E2E TEST PASSED! 🎉           ")
    else:
        print("           ❌ E2E TEST FAILED             ")
    print("===========================================")
    return test_passed

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    success = e2e_test_flow()
    sys.exit(0 if success else 1)
