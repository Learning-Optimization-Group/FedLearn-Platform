# FedLearn Platform

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://reactjs.org/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3-6DB33F?logo=springboot&logoColor=white)](https://spring.io/projects/spring-boot)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-EC2-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/)

**A full-stack, production-ready federated learning platform with a custom-built framework, web dashboard, and containerized client deployment.**

Built from scratch by the Learning Optimization Group at Rochester Institute of Technology under Professor Haibo Yang.

---

## 🌟 Overview

FedLearn Platform is an **open-source**, end-to-end solution for federated learning that combines:

- **Custom FL Framework** - Built from the ground up (not Flower-based) with advanced features like parameter chunking and parallel heartbeat mechanisms
- **Web Dashboard** - Modern React interface for managing projects, monitoring training, and viewing real-time logs
- **REST API** - Spring Boot backend with JWT authentication and WebSocket streaming
- **Docker Clients** - Pre-packaged containers for zero-installation client deployment
- **Production Deployment** - Running on AWS EC2 with PostgreSQL database

### Key Innovations

🔥 **Parameter Chunking** - Handles models >300MB by automatically chunking parameters during gRPC transmission

⚡ **Parallel Heartbeat** - Dual gRPC stub architecture prevents server timeout during long training sessions

🛡️ **DeComFL Integration** - Byzantine-robust aggregation for secure federated learning

🚀 **Full-Stack Integration** - Seamless orchestration from React UI → Spring Boot → Python FL Server → Docker Clients

---

## 🏗️ Architecture

<p align="center">
  <img src="architecture.svg" alt="FedLearn Platform Architecture" style="width: 100%; max-width: 1400px;">
</p>

### System Components

| Component | Technology | Purpose | Deployment |
|-----------|------------|---------|------------|
| **Frontend** | React 19 + Vite | Web dashboard, real-time logs | Vercel |
| **Backend API** | Spring Boot 3 | REST API, authentication, orchestration | AWS EC2 |
| **Database** | PostgreSQL | User data, projects, results | AWS EC2 |
| **FL Framework** | Python 3.10 + PyTorch | Custom federated learning server | AWS EC2 (spawned) |
| **FL Clients** | Docker + Python | Containerized training clients | Distributed |

### Data Flow

```
1. User creates project in React Dashboard
           ↓
2. Frontend sends REST API request to Spring Boot
           ↓
3. Spring Boot saves project to PostgreSQL
           ↓
4. Spring Boot spawns Python FL Server (via ProcessBuilder)
           ↓
5. FL Server starts gRPC server on dynamic port
           ↓
6. Docker/Native clients connect via gRPC
           ↓
7. Training begins with chunked parameter transfer
           ↓
8. Parallel heartbeat keeps connection alive
           ↓
9. Server logs streamed to React via WebSocket
           ↓
10. Results saved to PostgreSQL and displayed in UI
```

---

## 🚀 Key Features

### 1. Custom Federated Learning Framework

Built entirely from scratch without relying on existing FL frameworks like Flower.

**Capabilities**:
- FedAvg (Federated Averaging) aggregation
- DeComFL (Decomposed Federated Learning) with Byzantine robustness
- Support for CNNs, Transformers, and LLMs
- Non-IID data partitioning via Dirichlet distribution
- Mixed precision training
- Learning rate scheduling

**See**: [`framework/README.md`](framework/README.md)

---

### 2. Parameter Chunking for Large Models

**Challenge**: Models like LLMs can exceed 300MB, causing gRPC transmission failures.

**Solution**: Automatic parameter chunking during serialization.

```python
# Automatically chunks parameters >300MB
if model_size > 300_000_000:  # 300MB threshold
    chunks = chunk_parameters(parameters)
    for chunk in chunks:
        send_chunk(chunk)
```

**Benefits**:
- Supports large language models (OPT-125M, GPT-2, etc.)
- Memory-efficient transmission
- Transparent to end users

---

### 3. Parallel Heartbeat Mechanism

**Challenge**: During local training, clients cannot respond to server pings → connection timeout.

**Solution**: Dual gRPC stub architecture.

```
Client has TWO gRPC stubs:

Stub 1 (Training):          Stub 2 (Heartbeat):
- Send/receive parameters   - Send periodic pings
- Blocked during training   - Always responsive
- Heavy operations          - Lightweight
```

**Implementation**:
```python
# Training stub (blocking during fit)
training_stub.get_parameters()  # Blocked for minutes

# Heartbeat stub (parallel thread)
while training:
    heartbeat_stub.ping()  # Responds immediately
    time.sleep(1)
```

**Benefits**:
- Prevents false timeouts
- Supports long training sessions (hours)
- Maintains connection stability

---

### 4. Real-Time WebSocket Log Streaming

Live server logs displayed in React dashboard via STOMP/WebSocket.

```javascript
// Frontend subscribes to logs
client.subscribe(`/topic/logs/${projectId}`, (message) => {
    console.log(message.body);  // Real-time log line
});
```

**Backend streams Python process output**:
```java
// Spring Boot captures Python stdout
BufferedReader reader = new BufferedReader(
    new InputStreamReader(process.getInputStream())
);
String line;
while ((line = reader.readLine()) != null) {
    webSocketService.sendLogs(projectId, line);  // Broadcast via WebSocket
}
```

---

### 5. Docker-Based Client Deployment

Pre-packaged Docker images with framework + dependencies.

**User workflow**:
```bash
# 1. Pull Docker image
docker pull your-registry/fedlearn-client:latest

# 2. Run client (zero installation)
docker run -v /data:/data \
  fedlearn-client:latest \
  --server-address server.com:50051 \
  --client-id 0
```

**Benefits**:
- No Python/PyTorch installation required
- Consistent environment across clients
- Easy distribution to non-technical users

**See**: [`client-docker/README.md`](client-docker/README.md)

---

### 6. Full JWT Authentication & Authorization

Secure REST API with Spring Security + JWT tokens.

**Flow**:
```
1. User logs in → Spring Boot validates credentials
2. JWT token generated and returned
3. Frontend stores token in localStorage
4. All API requests include token in Authorization header
5. Spring Boot validates token on each request
6. User can only access their own projects
```

---

## 📊 Technology Stack

### Frontend
- **React 19** - Modern UI library
- **Vite 6** - Fast build tool
- **React Router v7** - Client-side routing
- **Axios** - HTTP client
- **STOMP.js** - WebSocket client
- **React Icons** - Icon library
- **Deployment**: Vercel

### Backend
- **Spring Boot 3** - Java framework
- **Spring Security** - Authentication/authorization
- **JWT** - Token-based auth
- **WebSocket (STOMP)** - Real-time communication
- **JPA/Hibernate** - ORM
- **PostgreSQL** - Relational database
- **Deployment**: AWS EC2 (Ubuntu)

### FL Framework
- **Python 3.10+** - Programming language
- **PyTorch 2.0+** - Deep learning framework
- **gRPC** - RPC framework
- **Protocol Buffers** - Serialization
- **NumPy** - Numerical computing
- **Transformers** - HuggingFace library (for LLMs)

### DevOps
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration
- **AWS EC2** - Cloud hosting
- **GitHub Actions** - CI/CD (optional)
- **Nginx** - Reverse proxy (optional)

---

## 📁 Repository Structure

```
FedLearn-Platform/
├── framework/                  # Custom FL framework (Python)
│   ├── src/fedlearn/          # Core package
│   │   ├── client/            # Client implementations
│   │   ├── server/            # Server and strategies
│   │   ├── communication/     # gRPC + serialization
│   │   ├── data/              # Data utilities
│   │   └── estimators/        # DeComFL estimators
│   ├── examples/              # Example applications
│   │   ├── simple_federation/ # MNIST + CNN
│   │   ├── llm_federation/    # OPT-125M fine-tuning
│   │   └── ecg_federation/    # ECG classification
│   ├── setup.py               # Pip installable
│   └── README.md              # Framework documentation
│
├── frontend/                   # React web application
│   ├── src/
│   │   ├── components/        # Reusable components
│   │   ├── pages/             # Page components
│   │   ├── services/          # API services
│   │   └── context/           # React Context (Auth)
│   ├── package.json
│   └── README.md              # Frontend documentation
│
├── backend/                    # Spring Boot API
│   └── fl-platform-api/
│       ├── src/main/java/com/federated/fl_platform_api/
│       │   ├── config/        # Security, WebSocket
│       │   ├── controller/    # REST endpoints
│       │   ├── service/       # Business logic
│       │   ├── repository/    # JPA repositories
│       │   ├── model/         # Entities
│       │   ├── security/      # JWT provider
│       │   └── flower/        # FlowerServerManager
│       ├── src/main/resources/
│       │   └── scripts/       # Python FL server scripts
│       └── README.md          # Backend documentation
│
├── client-docker/              # Docker client package
│   ├── fedlearn/              # Framework copy
│   ├── scripts/               # Client scripts
│   ├── Dockerfile             # Image definition
│   ├── requirements.txt       # Python dependencies
│   └── README.md              # Docker documentation
│
├── architecture.svg            # System architecture diagram
├── README.md                   # This file
└── LICENSE                     # Apache 2.0 license
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Java 17+**
- **Node.js 18+**
- **PostgreSQL 12+**
- **Docker** (for client deployment)

### 1. Setup Framework

```bash
cd framework
pip install -e .
```

**Documentation**: [`framework/README.md`](framework/README.md)

### 2. Setup Backend

```bash
cd backend/fl-platform-api

# Configure database in application.properties
# spring.datasource.url=jdbc:postgresql://localhost:5432/fedlearn_db

mvn spring-boot:run
```

**Documentation**: [`backend/fl-platform-api/README.md`](backend/fl-platform-api/README.md)

### 3. Setup Frontend

```bash
cd frontend
npm install
npm run dev
```

**Documentation**: [`frontend/README.md`](frontend/README.md)

### 4. Run FL Client (Docker)

```bash
cd client-docker
docker build -t fedlearn-client:latest .

docker run -v /data:/data \
  fedlearn-client:latest \
  --server-address localhost:50051 \
  --client-id 0
```

**Documentation**: [`client-docker/README.md`](client-docker/README.md)

---

## 📖 Documentation

Comprehensive documentation for each component:

| Component | Documentation |
|-----------|---------------|
| **FL Framework** | [`framework/README.md`](framework/README.md) |
| **Frontend** | [`frontend/README.md`](frontend/README.md) |
| **Backend API** | [`backend/fl-platform-api/README.md`](backend/fl-platform-api/README.md) |
| **Docker Client** | [`client-docker/README.md`](client-docker/README.md) |

### Developer Guides

- **Framework Development**: [`framework/CONTRIBUTING.md`](framework/CONTRIBUTING.md)
- **Frontend Development**: [`frontend/DEVELOPMENT.md`](frontend/DEVELOPMENT.md)
- **Backend Development**: [`backend/fl-platform-api/DEVELOPMENT.md`](backend/fl-platform-api/DEVELOPMENT.md)

---

## 🔬 Research & Publications

This platform implements algorithms from:

**DeComFL: Decomposed Federated Learning with Byzantine-Robust Aggregation**
- Authors: Haibo Yang, et al.
- Institution: Rochester Institute of Technology
- Implementation: [`framework/src/fedlearn/estimators/`](framework/src/fedlearn/estimators/)

### Citation

If you use FedLearn Platform in your research, please cite:

```bibtex
@article{yang2024decomfl,
  title={DeComFL: Decomposed Federated Learning},
  author={Yang, Haibo and [Co-authors]},
  journal={[Journal/Conference]},
  year={2024},
  institution={Rochester Institute of Technology}
}
```

---

## 🎯 Use Cases

### 1. Healthcare
- Train medical diagnosis models across hospitals
- Preserve patient privacy
- Aggregate knowledge without sharing sensitive data

### 2. Finance
- Fraud detection across banks
- Credit risk modeling
- Regulatory compliance (GDPR, HIPAA)

### 3. IoT & Edge Computing
- Distributed sensor networks
- Mobile device training (smartphones)
- Low-bandwidth environments

### 4. Research
- Academic federated learning experiments
- Algorithm benchmarking
- Privacy-preserving ML research

---

## 🛡️ Security & Privacy

### Data Privacy
- ✅ Raw data never leaves client devices
- ✅ Only model updates transmitted
- ✅ Differential privacy support (optional)
- ✅ Secure aggregation algorithms

### Authentication
- ✅ JWT-based user authentication
- ✅ Project-level access control
- ✅ Secure WebSocket connections

### Network Security
- ✅ TLS/SSL for gRPC (configurable)
- ✅ CORS configuration
- ✅ Input validation & sanitization

---

## 🚀 Deployment

### Development (Local)

```bash
# Terminal 1: Backend
cd backend/fl-platform-api && mvn spring-boot:run

# Terminal 2: Frontend
cd frontend && npm run dev

# Terminal 3: FL Client
cd framework/examples/simple_federation
python run_server.py  # Server
python run_client.py --id 0  # Client
```

### Production (AWS EC2)

**Backend + FL Server**:
- AWS EC2 instance (Ubuntu 22.04)
- PostgreSQL installed locally
- Spring Boot as systemd service
- Python FL servers spawned dynamically

**Frontend**:
- Deployed on Vercel
- Automatic deployments from `main` branch

**Configuration**:
```bash
# Backend environment variables
DATABASE_URL=jdbc:postgresql://localhost:5432/fedlearn_db
JWT_SECRET=your-secret-key
FL_SCRIPTS_PATH=/path/to/scripts
```

---

## 🤝 Contributing

We welcome contributions! This is an open-source project under Apache 2.0 license.

### How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

See individual component documentation:
- Framework: [`framework/CONTRIBUTING.md`](framework/CONTRIBUTING.md)
- Frontend: [`frontend/DEVELOPMENT.md`](frontend/DEVELOPMENT.md)
- Backend: [`backend/fl-platform-api/DEVELOPMENT.md`](backend/fl-platform-api/DEVELOPMENT.md)

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on collaboration
- Help newcomers

---

## 📝 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

```
Copyright 2024 Learning Optimization Group, Rochester Institute of Technology

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 👥 Team

**Principal Investigator**: Professor Haibo Yang  
**Institution**: Rochester Institute of Technology  
**Research Group**: Learning Optimization Group

**Developer**: Chinmay (MS Computer Science, RIT)

---

## 🙏 Acknowledgments

- Rochester Institute of Technology for research support
- Learning Optimization Group for collaboration
- Open-source community for inspiration

---

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/Learning-Optimization-Group/FedLearn-Platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Learning-Optimization-Group/FedLearn-Platform/discussions)
- **Email**: haibo.yang@rit.edu (Professor Haibo Yang)

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐️ on GitHub!

---

**Built with ❤️ by the Learning Optimization Group at Rochester Institute of Technology**

**Open Source • Production Ready • Research Grade**