# FedLearn Platform

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://reactjs.org/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3-6DB33F?logo=springboot&logoColor=white)](https://spring.io/projects/spring-boot)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-EC2-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com/)

**A full-stack, federated learning platform with a custom-built framework, web dashboard, and containerized client deployment.**

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

📉 **DeComFL Integration** - Dimension-free communication: O(1) scalar values per round via zeroth-order optimization

🚀 **Full-Stack Integration** - Seamless orchestration from React UI → Spring Boot → Python FL Server → Docker Clients

---

## 🏗️ Architecture

![FedLearn Platform Architecture](architecture.png)

### System Components

| Component              | Technology               | Purpose                                  | Deployment                                          |
| ---------------------- | ------------------------ | ---------------------------------------- | --------------------------------------------------- |
| **Frontend**     | React 19 + Vite + TS     | Web dashboard, real-time telemetry       | Local Vite (`:5173`) or static bundle               |
| **Backend API**  | Spring Boot 3 (Java 21)  | REST + STOMP, auth, FL-server lifecycle  | AWS EC2 behind nginx + Let's Encrypt                |
| **Database**     | H2 (file-mode)           | Users, projects, training results        | EBS-backed on the same EC2; PostgreSQL is `production` profile (unfinished) |
| **FL Framework** | Python 3.10 + PyTorch    | Custom federated learning server         | Spawned by backend via `ProcessBuilder`             |
| **FL Clients**   | Docker + Python          | Containerized training clients           | Heterogeneous: Jetson AGX Orin / M4 Max / Zephyrus  |
| **Desktop**      | Electron + TS + dockerode | Host-side orchestrator for FL clients   | Packaged for macOS / Linux / Windows (CPU + CUDA)   |

### Data Flow

```
Browser
  → nginx :443 (TLS, Let's Encrypt) — only on EC2; local dev hits :8081 direct
  → Spring Boot REST + STOMP (:8081, loopback-only on EC2)
  → H2 / PostgreSQL  (project + user state)
  → spawns Python FL server via ProcessBuilder
  → FL server gRPC on a dynamic port in :50000-50010
  → FL clients (Docker / native) connect over gRPC
       ↘ training stub (long blocking calls)
       ↘ heartbeat stub (parallel thread, keeps connection alive)
  → server stdout streamed back as STOMP messages → live in the React dashboard
  → round results persisted, surfaced as sparklines + telemetry
```

Live demo deployment: **https://fedlearn.duckdns.org** (`ec2demo` Spring profile). See `docs/guides/aws_deployment_guide.md`.

---

## 🚀 Key Features

### 1. Custom Federated Learning Framework

Built entirely from scratch without relying on existing FL frameworks like Flower.

**Capabilities**:

- FedAvg (Federated Averaging) aggregation
- DeComFL (Dimension-Free Communication FL) — O(1)-per-round communication via zeroth-order optimization
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

### 6. Stateless JWT via HttpOnly Cookies

Spring Security signs a stateless JWT and delivers it to the browser as an **HttpOnly, Secure, SameSite-tightened cookie**. The frontend never sees the token in JavaScript — `withCredentials: true` on Axios is the only thing it does to authenticate.

**Flow**:

```
1. User logs in   → Spring Boot validates credentials, signs a JWT
2. Backend sets jwtToken as an HttpOnly cookie in the response
3. Browser auto-sends the cookie on every subsequent request
4. JwtAuthenticationFilter reads the cookie, validates, sets SecurityContext
5. Resource-level checks ensure users only see their own projects
```

This deliberately closes the XSS exfiltration vector: there is no `localStorage` or JS-readable token to steal. The same model applies to the Electron desktop app — auth state lives in the main-process session, never crosses into the renderer.

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

- **Spring Boot 3** + **Java 21** + **Gradle**
- **Spring Security** + **JWT** delivered as HttpOnly cookies
- **WebSocket (STOMP)** for live log + telemetry streaming
- **JPA / Hibernate** (validate-only) — schema owned by **Flyway**
- **H2** (file-mode) on the EC2 demo; **PostgreSQL** wired in the `production` profile
- **Deployment**: AWS EC2 (Ubuntu) behind **nginx** + **Let's Encrypt**

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
├── architecture.png            # System architecture diagram
├── README.md                   # This file
└── LICENSE                     # Apache 2.0 license
```

---

## 🚀 Quick Start

### Prerequisites

- **Java 21**
- **Node.js 18+**
- **Python 3.10+** (only if you run the FL framework directly; the Docker client bundles its own runtime)
- **Docker** (for FL clients)

H2 is file-mode in dev — no PostgreSQL needed locally.

### Run the full stack

```bash
./launch_all.sh
```

This opens four terminal windows: backend on `:8081` (Spring profile `dev`), Vite on `:5173`, Electron on `:9000`, and the FL-client launcher.

### Run components individually

```bash
# Backend
cd backend/fl-platform-api
SPRING_PROFILES_ACTIVE=dev ./gradlew bootRun

# Frontend — three modes, all mirror Spring profiles 1:1
cd frontend && npm install
npm run dev               # full-local: backend on localhost:8081
npm run dev:ec2demo       # frontend-local, backend on https://fedlearn.duckdns.org via Vite proxy
npm run build             # production bundle

# FL framework (Python)
cd framework
pip install -e .

# Docker FL client
cd client-docker
docker build -t fedlearn-client:latest .
docker run -v /data:/data fedlearn-client:latest --server-address localhost:50051 --client-id 0
```

For deployed environments, see **`docs/guides/aws_deployment_guide.md`** (the canonical EC2 deploy procedure) and **`client-docker/DEPLOYMENT_GUIDE.md`** (Jetson and native clients).

---

## 📖 Documentation

Comprehensive documentation for each component:

| Component               | Documentation                                                           |
| ----------------------- | ----------------------------------------------------------------------- |
| **FL Framework**  | [`framework/README.md`](framework/README.md)                             |
| **Frontend**      | [`frontend/README.md`](frontend/README.md)                               |
| **Backend API**   | [`backend/fl-platform-api/README.md`](backend/fl-platform-api/README.md) |
| **Docker Client** | [`client-docker/README.md`](client-docker/README.md)                     |

**Cross-cutting docs** (full map: [`docs/README.md`](docs/README.md)):

- **Research papers ↔ implementation**: [`docs/research/papers-and-implementation.md`](docs/research/papers-and-implementation.md)
- **Design system (Instrument)**: [`docs/design/instrument-design-system.md`](docs/design/instrument-design-system.md)
- **v2 build status & architecture**: [`docs/v2/STATUS.md`](docs/v2/STATUS.md) · **Deep-dive wikis**: [`docs/wikis/`](docs/wikis/)

### Operational Guides

- **AWS deployment**: [`docs/guides/aws_deployment_guide.md`](docs/guides/aws_deployment_guide.md)
- **Local + RIT lab deployment**: [`docs/guides/local_and_rit_deployment_guide.md`](docs/guides/local_and_rit_deployment_guide.md)
- **Pneumonia federation demo plan**: [`docs/guides/pneumonia_demo_plan.md`](docs/guides/pneumonia_demo_plan.md)
- **AWS / Electron architectural review**: [`docs/guides/aws_and_electron_architecture_risks.md`](docs/guides/aws_and_electron_architecture_risks.md)
- **AWS audit (Tier 2 backlog)**: [`docs/guides/AWS_AUDIT.md`](docs/guides/AWS_AUDIT.md)
- **Framework contribution guide**: [`framework/CONTRIBUTING.md`](framework/CONTRIBUTING.md)

---

## 🔬 Research & Publications

This platform is grounded in published research. **Full papers-to-code mapping** (DeComFL, FoT, HiSo, DPZV — what's implemented vs. on the roadmap): [`docs/research/papers-and-implementation.md`](docs/research/papers-and-implementation.md).

**Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization** (ICLR 2025)

- Authors: Zhe Li, Bicheng Ying, Zidong Liu, Chaosheng Dong, Haibo Yang (Rochester Institute of Technology)
- Paper: [arXiv:2405.15861](https://arxiv.org/abs/2405.15861) · Reference implementation: [ZidongLiu/DeComFL](https://github.com/ZidongLiu/DeComFL) (Apache-2.0)
- Implementation: [`framework/src/fedlearn/estimators/`](framework/src/fedlearn/estimators/)

**Federation over Text (FoT)** — additive, local-LLM-only text-federation research mode (orthogonal to the gradient path)

- Reference: [arXiv:2604.16778](https://arxiv.org/abs/2604.16778) · Implementation: [`framework/src/fedlearn/fot/`](framework/src/fedlearn/fot/)

### Citation

If you use FedLearn Platform in your research, please cite the DeComFL paper:

```bibtex
@inproceedings{li2025decomfl,
  title={Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization},
  author={Li, Zhe and Ying, Bicheng and Liu, Zidong and Dong, Chaosheng and Yang, Haibo},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
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
- ✅ Only model updates (FedAvg) or O(1) gradient scalars + seeds (DeComFL) are transmitted

### Authentication

- ✅ Stateless JWT delivered as **HttpOnly + Secure** cookies (no JS-readable token storage)
- ✅ Resource-level authorization (users only see their own projects)
- ✅ STOMP WebSocket auth via the same cookie

### Network Security

- ✅ TLS terminated at nginx (Let's Encrypt) on the EC2 deployment
- ✅ Backend `:8081` bound to `127.0.0.1` only — no public side-door
- ✅ Strict CORS allowlist — Spring fails fast on missing config
- ⚠️ gRPC FL client traffic is currently plaintext over WAN (audit item #37)

---

## 🚀 Deployment

### Local development

`./launch_all.sh` launches everything in parallel terminal windows. Or run individually:

```bash
# Backend (Gradle, Java 21)
cd backend/fl-platform-api && SPRING_PROFILES_ACTIVE=dev ./gradlew bootRun

# Frontend
cd frontend && npm run dev

# FL example
cd framework/examples/simple_federation
python run_server.py
python run_client.py --id 0
```

### EC2 demo (`ec2demo` profile)

Live at **https://fedlearn.duckdns.org**. Deploy procedure: [`docs/guides/aws_deployment_guide.md`](docs/guides/aws_deployment_guide.md).

- AWS EC2 (Ubuntu 24.04 LTS, `r5.large`)
- nginx terminates TLS on `:443`, proxies to Spring Boot on `127.0.0.1:8081`
- Let's Encrypt certbot for auto-renewing TLS
- H2 file-mode at `~/app/data/`, EBS-backed across reboots
- Spring Boot as a systemd service (`fedlearn.service`)
- Python FL servers spawned by `FlowerServerManager`

Required env vars (set in `/etc/systemd/system/fedlearn.service`):

```bash
APP_JWT_SECRET=<openssl rand -base64 64>
APP_INTERNAL_API_KEY=<openssl rand -hex 32>
CORS_ALLOWED_ORIGINS=https://fedlearn.duckdns.org,http://localhost:5173
APP_AUTH_COOKIE_SECURE=true
```

### ECS Fargate (`production` profile)

Wired but unfinished. Tier 2 audit items 10–17 in [`docs/guides/AWS_AUDIT.md`](docs/guides/AWS_AUDIT.md) describe what's missing (S3 model storage, FL servers as ECS tasks, multi-replica safety).

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
- Backend: [`backend/fl-platform-api/DEVELOPMENT.md`](backend/fl-platform-api/DEVELOPMENT.md)
- Frontend: [`frontend/README.md`](frontend/README.md) and `frontend/.env.example`

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
