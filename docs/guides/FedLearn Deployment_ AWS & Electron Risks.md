# **FedLearn Platform Architectural Review: AWS Migration and Electron Client Deployment**

## **Executive Summary and Architectural Context**

The FedLearn platform represents a sophisticated implementation of privacy-preserving machine learning, currently structured around a Three-Tier architecture comprising a Spring Boot Application Programming Interface (API), a dedicated Service layer, a Repository layer, and a PostgreSQL database. Federated Learning (FL) orchestration relies on dynamic process management, wherein a Java-based FlowerServerManager allocates ephemeral network ports and spawns local Python processes (fl\_server.py) to handle discrete training rounds. The Communication Tier utilizes gRPC operating over the HTTP/2 protocol for transmitting PyTorch model weights, with WebSocket connections streaming real-time execution logs back to the central control plane. Distributed client nodes operate within containerized Docker environments tailored for localized edge hardware, bridging both standard discrete Graphics Processing Units (GPUs) and integrated System-on-Chip (SoC) platforms such as the NVIDIA Jetson.

The proposed architectural transition entails migrating the centralized backend infrastructure—encompassing the Spring Boot API, PostgreSQL database, and Python FL Servers—to Amazon Web Services (AWS), while concurrently deploying the client-side environment as a bundled Electron desktop application. This comprehensive analysis evaluates the distributed systems concepts underlying the framework, identifies structural bottlenecks, and delineates the pragmatic architectural refactoring required to execute this deployment securely and efficiently. The analysis explores AWS dynamic orchestration, Wide Area Network (WAN) gRPC stability, PyTorch tensor serialization optimizations, and Electron process security, ensuring the platform scales reliably under production workloads.

## **1\. AWS Orchestration: Migrating Dynamic Port Allocation to Containerized Task Scheduling**

### **1.1 Conceptual Foundation: Process Spawning Versus Cloud-Native Orchestration**

In monolithic or single-server environments, managing secondary application runtimes—such as a Spring Boot Java application launching a Python machine learning script—is traditionally achieved via operating system-level process forking. The parent process dynamically queries the kernel for an available ephemeral port, typically by binding a network socket to port zero. The kernel responds by assigning an available port, which the parent records and subsequently passes as an execution argument to the child process. This pattern tightly couples the parent and child processes to the same underlying physical or virtual host, sharing the same network namespace, file descriptors, and resource pools.

When transitioning to a cloud-native, horizontally scaled environment such as AWS Elastic Container Service (ECS) or Amazon Elastic Kubernetes Service (EKS), this paradigm transforms into a severe architectural anti-pattern. Containerized environments enforce strict resource constraints via Linux control groups (cgroups) and isolate execution contexts using kernel namespaces. A Spring Boot container that attempts to spawn intensive Python machine learning workloads internally will rapidly induce resource starvation, leading to Out-Of-Memory (OOM) container terminations and critical horizontal scaling failures. Furthermore, dynamically allocating ports within a single container isolates that port from the broader Virtual Private Cloud (VPC) network unless specifically mapped and registered at the infrastructure load balancer level.1

### **1.2 Codebase Analysis: The FlowerServerManager.java Implementation**

An examination of the FedLearn codebase reveals that the FlowerServerManager.java class currently manages the federated server lifecycle utilizing the native Java ProcessBuilder.3 The application binds a temporary ServerSocket to port zero, retrieves the local port assigned by the host operating system, and invokes the fl\_server.py script via a shell command. During this invocation, it passes critical configuration parameters such as \--project-id, \--strategy, and \--port.

In an AWS environment, if the Spring Boot application is containerized and scaled behind an Application Load Balancer (ALB), attempting to execute fl\_server.py internally violates the immutability and single-responsibility principles of modern container design. The ports allocated internally by the Java process will not be exposed to the ALB or registered with any external Domain Name System (DNS), rendering the newly spawned federated learning server completely unreachable by external Edge clients.4 The dynamic port mapping strategies natively supported by AWS ECS—which allow the host port to be dynamically chosen from the ephemeral port range—require the container port to be statically defined in the task definition, fundamentally conflicting with the runtime generation of ports performed by the current Java implementation.5

### **1.3 Architectural Optimization: ECS RunTask API and Service Connect Discovery**

To align with modern distributed systems practices, the backend must transition from operating system-level process management to infrastructure-level task orchestration. Instead of using ProcessBuilder to launch a local script, the Spring Boot API must leverage the AWS Software Development Kit (SDK) to invoke the ECS RunTask API.7 This approach launches the federated learning server as a completely independent container—for example, utilizing the serverless AWS Fargate compute engine—equipped with its own dedicated CPU, memory allocation, and Elastic Network Interface (ENI) operating in awsvpc network mode.8

For dynamic port resolution and routing within this ephemeral landscape, AWS ECS Service Connect provides an exceptionally robust solution. Powered by Envoy proxy sidecars, Service Connect automatically manages service discovery without relying on legacy DNS Time-To-Live (TTL) propagation delays or complex Load Balancer target group manipulations.9 When the RunTask API is invoked, the newly provisioned fl\_server.py container registers its private IP address and port within an AWS Cloud Map namespace. This registration allows the system to route traffic based on logical identifiers, such as fl-server-\<project-id\>, rather than relying on brittle static IP and port combinations.11

The following table illustrates the architectural shift required for the orchestration logic:

| Architectural Dimension | Current Architecture (ProcessBuilder) | Proposed Architecture (AWS RunTask) |
| :---- | :---- | :---- |
| **Compute Isolation** | Shared CPU/Memory with Java Parent | Dedicated AWS Fargate Task per Project |
| **Network Namespace** | Localhost (Shared Container Network) | Isolated Elastic Network Interface (awsvpc) |
| **Port Allocation** | Runtime ServerSocket Query | ECS Automated Cloud Map Registration |
| **Traffic Routing** | Direct Host IP/Port Connection | Envoy Proxy via ECS Service Connect |
| **Scaling Limitations** | Constrained by Single Host Resources | Horizontally Scalable across AWS Regions |

### **1.4 Code Refactoring: Transitioning to AWS SDK Task Invocation**

The following refactoring demonstrates the modernization of the server initialization logic, moving from local command-line execution to structured AWS ECS orchestration. The refactored code completely eliminates the need to allocate ports manually, delegating network routing to the AWS infrastructure.

**Before: Local Process Invocation Anti-Pattern (FlowerServerManager.java)**

Java

// Anti-pattern for cloud deployment: Local port binding and process spawning  
public void startServerForProject(String projectId, String modelPath, String strategy) {  
    try (ServerSocket s \= new ServerSocket(0)) {  
        int dynamicPort \= s.getLocalPort();  
        // The Java application spawns a child process internally  
        ProcessBuilder pb \= new ProcessBuilder(  
            "python", "fl\_server.py",   
            "--project-id", projectId,  
            "--port", String.valueOf(dynamicPort),  
            "--strategy", strategy  
        );  
        Process process \= pb.start();  
        activeProcesses.put(projectId, process);  
    } catch (IOException e) {  
        log.error("Failed to start FL Server", e);  
    }  
}

**After: Cloud-Native ECS Orchestration via AWS SDK (FlowerServerManager.java)**

Java

import software.amazon.awssdk.services.ecs.EcsClient;  
import software.amazon.awssdk.services.ecs.model.\*;

// Optimized: Launching an independent AWS Fargate Task per FL Project  
public void startServerForProject(String projectId, String modelPath, String strategy) {  
    // Port mapping is handled internally by ECS Service Connect / awsvpc mode  
    EcsClient ecsClient \= EcsClient.builder().build();

    // Dynamically override the container execution command  
    ContainerOverride containerOverride \= ContainerOverride.builder()  
       .name("fl-server-container")  
       .command("python", "fl\_server.py",   
                 "--project-id", projectId,   
                 "--strategy", strategy)  
       .build();

    TaskOverride taskOverride \= TaskOverride.builder()  
       .containerOverrides(containerOverride)  
       .build();

    // Construct the API request to launch a serverless compute task  
    RunTaskRequest runTaskRequest \= RunTaskRequest.builder()  
       .cluster("fedlearn-ecs-cluster")  
       .taskDefinition("fl-server-task-definition")  
       .launchType(LaunchType.FARGATE)  
       .overrides(taskOverride)  
       .networkConfiguration(NetworkConfiguration.builder()  
           .awsvpcConfiguration(AwsVpcConfiguration.builder()  
               .subnets("subnet-xxxxxxxx", "subnet-yyyyyyyy")  
               .securityGroups("sg-zzzzzzzz")  
               .assignPublicIp(AssignPublicIp.ENABLED)  
               .build())  
           .build())  
       .build();

    try {  
        RunTaskResponse response \= ecsClient.runTask(runTaskRequest);  
        log.info("Provisioned ECS Task for Project ID: {}", projectId);  
        // Store the immutable Task ARN instead of the ephemeral local Process object  
        activeTasks.put(projectId, response.tasks().get(0).taskArn());  
    } catch (EcsException e) {  
        log.error("Failed to provision ECS Task", e);  
    }  
}

## **2\. Wide Area Network Latency and gRPC Communication Reliability**

### **2.1 Conceptual Foundation: HTTP/2 Connection State and Middlebox Interference**

The gRPC framework fundamentally relies upon HTTP/2 as its underlying transport layer. HTTP/2 is designed around the concept of long-lived, multiplexed Transmission Control Protocol (TCP) connections, which allow multiple concurrent streams to be transmitted over a single socket.12 In a Local Area Network (LAN) or a standard Docker bridge network environment, idle connections can remain open indefinitely without interference. However, deploying the framework across a Wide Area Network (WAN) via AWS introduces intermediate routing hardware—specifically, Network Address Translation (NAT) gateways, stateful firewalls, and Load Balancers.

These intermediate middleboxes are programmed to aggressively cull idle connections to conserve memory and routing table space. AWS Application Load Balancers (ALB) enforce a strict default idle timeout of 60 seconds, while AWS Network Load Balancers (NLB) enforce a slightly more permissive 350-second idle timeout.13 In a federated learning context, if a remote client requires several minutes to execute a local PyTorch training phase—for example, iterating through multiple epochs of a large proprietary dataset—no data is transmitted over the gRPC channel during this compute phase. Consequently, the AWS infrastructure will silently drop the connection. When the client subsequently attempts to submit its updated model weights, it will encounter a fatal GOAWAY frame or an abrupt RST\_STREAM error, necessitating a full reconnection protocol and potentially corrupting the federated training round.14

### **2.2 Codebase Analysis: The Limitations of Application-Layer Heartbeats**

The FedLearn codebase attempts to mitigate connection loss through an application-layer heartbeat mechanism. The GrpcClient within grpc\_client.py establishes two distinct network channels: a primary data channel for model transfer and a dedicated heartbeat\_channel. A background threading loop executes the start\_heartbeat() function, which periodically invokes the Heartbeat() Remote Procedure Call (RPC) handler on the grpc\_servicer.py. This handler notifies the FLCoordinator object within coordinator.py of the client's current liveness state.3

While an application-layer heartbeat successfully informs the server's overarching business logic that the client is theoretically active, it fundamentally fails to prevent AWS middleboxes from terminating the primary data channel. Because the application logic splits the heartbeat and the model transfer into two separate TCP sockets, maintaining activity on the heartbeat\_channel does absolutely nothing to reset the idle timer on the primary channel. The primary data channel must be kept alive at the HTTP/2 transport layer via native gRPC Keepalive frames.

### **2.3 Architectural Optimization: HTTP/2 PING Frames and Transport-Layer Keepalive**

To stabilize long-lived streams during extended local training cycles, native gRPC Keepalive settings must be explicitly injected into the channel arguments during instantiation.15 Keepalive operates entirely beneath the application layer by transmitting an HTTP/2 PING frame at a mathematically defined interval.16

The most critical configuration parameter in this context is grpc.keepalive\_permit\_without\_calls. By default, gRPC implementations suppress PING frames if there are no active RPC streams in flight to prevent unintended denial-of-service behaviors against servers. However, during a federated training compute phase, no RPCs are actively streaming. Authorizing the transmission of PING frames even when no active RPCs are in flight actively resets the idle timers on AWS ALBs and NLBs, ensuring the primary channel remains open and ready to accept the massive payload of updated model weights once the compute phase concludes.13 Furthermore, setting the grpc.client\_idle\_timeout\_ms to a virtually infinite value prevents the gRPC library itself from proactively tearing down the connection.

### **2.4 Code Refactoring: Implementing Transport-Layer Resilience**

The following refactoring demonstrates how to implement transport-layer resilience for the client connection, specifically tailored for environments traversing an AWS Network Load Balancer subject to a 350-second timeout limit.

**Before: Default Insecure Channel Instantiation (grpc\_client.py)**

Python

import grpc

class GrpcClient:  
    def \_\_init\_\_(self, target\_address):  
        \# Default channel instantiation.  
        \# Highly susceptible to silent connection drops via AWS Load Balancers  
        \# due to the lack of transport-layer keepalive configurations.  
        self.channel \= grpc.insecure\_channel(target\_address)  
        self.heartbeat\_channel \= grpc.insecure\_channel(target\_address)

**After: Resilient Channel with Keepalive Parameters (grpc\_client.py)**

Python

import grpc

class GrpcClient:  
    def \_\_init\_\_(self, target\_address):  
        \# Keepalive parameters specifically tuned to prevent AWS NLB/ALB connection culling.  
        \# These parameters operate at the HTTP/2 frame level, invisible to the application logic.  
        keepalive\_options \=  
          
        self.channel \= grpc.insecure\_channel(target\_address, options=keepalive\_options)  
        self.heartbeat\_channel \= grpc.insecure\_channel(target\_address, options=keepalive\_options)

## **3\. PyTorch Data Flow Optimization: Chunked Streaming and Zero-Copy Serialization**

### **3.1 Conceptual Foundation: Memory Allocation and Serialization Overheads**

In a federated learning architecture, client nodes must regularly transmit the updated neural network weights—represented as the global state\_dict dictionary in PyTorch—back to the centralized coordinator for aggregation via algorithms such as FedAvg. Machine learning models, particularly contemporary Transformer architectures like the OPT-125M referenced in the framework's documentation, or complex Convolutional Neural Networks (CNNs) like DenseNet161, require substantial memory. A 100-Megabyte model serialized into memory can rapidly trigger Out-Of-Memory (OOM) exceptions if the serialization logic involves inefficient intermediate buffering.3

The gRPC protocol enforces a default maximum message size of 4 Megabytes. Attempting to transmit a large PyTorch model as a single unary RPC call will result in a fatal RESOURCE\_EXHAUSTED error. Consequently, the framework must divide the serialized byte array into discrete chunks and utilize gRPC's bidirectional stream primitive.17 However, memory management during this chunking phase is notoriously difficult in Python due to the constraints of the Global Interpreter Lock (GIL) and reference counting. If the entire model is first serialized into a monolithic byte string, and then subsequently sliced into a comprehensive list of smaller byte strings via standard Python array slicing (chunk \= data\[start:end\]), the memory requirement momentarily triples. The system must allocate memory for the original PyTorch tensor architecture, the initial monolithic serialized byte buffer, and the complete array of duplicated sliced chunks. On resource-constrained edge devices, this tripling of memory footprint causes instantaneous system failure.

### **3.2 Codebase Analysis: Model Submission Logic and Thresholds**

The codebase currently implements an adaptive submission logic paradigm within the grpc\_client.py module. A constant variable, STREAMING\_THRESHOLD\_MB, is established and set to 100 Megabytes. For models possessing a memory footprint beneath this limit, the system utilizes the \_submit\_update\_unary() function; for payloads exceeding this threshold, it intelligently defaults to the \_submit\_update\_stream() function.3

While the fundamental architectural distinction between small and large payload handling is conceptually correct, applying unary transmission for any payload approaching 100 Megabytes directly violates gRPC architectural best practices and guarantees exceeding the default 4MB HTTP/2 frame limits unless the channel arguments are aggressively and dangerously expanded. Furthermore, an analysis of typical streaming implementations reveals that building the entire chunk sequence as an in-memory list prior to initiating the transmission creates a severe performance bottleneck during Python garbage collection, leading to noticeable latency spikes before the network transmission even begins.19

### **3.3 Architectural Optimization: Zero-Copy Slicing and Generator-Based Streaming**

To optimize the client's memory footprint—an absolute operational necessity for edge devices operating on constrained hardware profiles like the NVIDIA Jetson—the streaming function must be architecturally refactored to utilize Python's yield statement.20 Constructing a generator function allows the underlying C++ gRPC library bindings to consume chunks of bytes on-demand. This lazy-evaluation approach prevents the application from storing the entire chunked payload array in memory simultaneously; chunks are generated, transmitted, and immediately marked for garbage collection.

Additionally, leveraging the standard io.BytesIO module avoids unnecessary and highly latent disk Input/Output operations, maintaining the serialization payload purely within Random Access Memory (RAM). When combined with Python's built-in memoryview object, it enables zero-copy slicing of the underlying buffer.21 A memoryview object allows the application to expose the internal data of the bytes array without creating a duplicate copy. Slicing a memoryview merely creates a lightweight pointer to the specific offset of the original buffer, entirely neutralizing the memory-tripling effect described previously.

The differences between the traditional and optimized memory handling approaches are contrasted in the following table:

| Memory Metric | Standard Slicing (bytes\[start:end\]) | Zero-Copy Slicing (memoryview) |
| :---- | :---- | :---- |
| **Data Duplication** | Yes (Creates a new bytes object per chunk) | No (Creates a pointer to the original buffer) |
| **Peak RAM Usage** | 3x Model Size | 1.1x Model Size |
| **Garbage Collection** | Heavy (Reclaiming thousands of byte objects) | Negligible (Pointer destruction is instantaneous) |
| **Execution Speed** | CPU-bound by memory allocation | Bound only by network throughput |

### **3.4 Code Refactoring: Optimizing Memory During Model Streaming**

The following refactoring demonstrates the precise implementation required to optimize the PyTorch to gRPC streaming pipeline, utilizing efficient memory views and Python generators to flatten the memory curve.

**Before: Inefficient Memory Buffering (grpc\_client.py)**

Python

import io  
import torch  
import grpc\_proto\_pb2

def \_submit\_update\_stream(self, state\_dict):  
    buffer \= io.BytesIO()  
    torch.save(state\_dict, buffer)  
    byte\_data \= buffer.getvalue()  
      
    \# Architectural Bottleneck: Highly inefficient slicing.  
    \# Creates a massive array of duplicated byte objects in memory simultaneously,  
    \# causing memory spikes that can crash edge devices.  
    chunks \=  
    chunk\_size \= 1024 \* 1024 \* 2 \# 2MB chunks  
    for i in range(0, len(byte\_data), chunk\_size):  
        chunk \= byte\_data\[i:i \+ chunk\_size\]  
        chunks.append(grpc\_proto\_pb2.ModelChunk(data=chunk))  
          
    \# Transmitting the pre-built list to the server  
    self.stub.SubmitModelUpdateStream(iter(chunks))

**After: Zero-Copy Yield Generator (grpc\_client.py)**

Python

import io  
import torch  
import grpc\_proto\_pb2

def \_generate\_model\_chunks(self, state\_dict):  
    buffer \= io.BytesIO()  
    \# Save the PyTorch model directly to the in-memory bytes buffer  
    torch.save(state\_dict, buffer)  
      
    \# Extract the total byte size and explicitly reset the buffer pointer to zero  
    buffer\_size \= buffer.tell()  
    buffer.seek(0)  
      
    \# Read the buffer into a zero-copy memoryview to prevent data duplication.  
    \# memoryview allows slicing the underlying bytes without creating new byte objects.  
    mem\_view \= memoryview(buffer.getbuffer())  
      
    chunk\_size \= 1024 \* 1024 \* 2 \# 2MB chunks to safely fit within gRPC 4MB limits  
      
    \# Yield chunks dynamically via a Python generator protocol.  
    \# The gRPC core consumes one chunk and transmits it before the next is sliced.  
    for offset in range(0, buffer\_size, chunk\_size):  
        yield grpc\_proto\_pb2.ModelChunk(  
            data=mem\_view\[offset:offset \+ chunk\_size\].tobytes()  
        )  
          
def \_submit\_update\_stream(self, state\_dict):  
    \# Pass the generator function directly to the gRPC streaming API.  
    \# Execution pauses until gRPC requests the next chunk, keeping RAM usage entirely flat.  
    self.stub.SubmitModelUpdateStream(self.\_generate\_model\_chunks(state\_dict))

## **4\. Electron Desktop Integration: Architecture, Feasibility, and Constraints**

### **4.1 Conceptual Foundation: The Inter-Process Communication Bridge and Native Execution**

Electron operates on a strict multi-process architecture consisting of a single Main Process and multiple Renderer Processes. The Main Process operates a full Node.js environment, possessing unrestricted access to the host operating system, filesystem, and native dependencies. The Renderer Processes, conversely, operate sandboxed Chromium browser instances responsible exclusively for rendering the user interface.23 Direct communication between the user interface and the underlying filesystem or operating system must traverse the Inter-Process Communication (IPC) bridge, a serialization boundary designed to prevent malicious web content from escalating privileges.24

The proposed architecture intends to bundle the PyTorch federated learning client environment directly into the Electron application, distributing it as a unified desktop application. Attempting to bundle a heavyweight machine learning environment—which encompasses the Python interpreter, PyTorch binaries, complex NVIDIA CUDA toolkits, and deeply nested C++ gRPC dependencies—directly into the Electron app.asar package presents profound scalability and distribution impossibilities.25 A typical PyTorch environment equipped with CUDA support effortlessly exceeds four Gigabytes in disk space. Bundling this massive payload renders standard desktop distribution mechanisms, such as Over-The-Air (OTA) Electron auto-updates or App Store deployment, entirely unviable due to bandwidth constraints and unpacking timeouts during installation.

### **4.2 Codebase Analysis: Docker Management on Disparate Hardware Variants**

A detailed review of the fedlearn\_client\_docker.txt deployment documentation highlights highly specific hardware execution requirements that further complicate a unified binary approach. Environments equipped with discrete GPUs, such as dedicated Linux or Windows workstations, utilize standard NVIDIA Container Toolkits, activated via flags such as docker run \--gpus all.3

Conversely, NVIDIA Jetson Edge devices possess integrated Tegra GPUs located directly on the System-on-Chip (SoC). The documentation explicitly prohibits the use of the \--runtime nvidia flag on Jetson platforms, noting that the standard toolkit searches the Linux kernel device tree for discrete Peripheral Component Interconnect Express (PCIe) metadata. Failing to find a discrete card, the container hangs indefinitely.3 Instead, Jetson deployments require precise, low-level device mounting commands (e.g., \--device /dev/nvhost-ctrl, \--device /dev/nvhost-ctrl-gpu) to pass the SoC GPU components directly into the container filesystem.3

Given these stark hardware disparities, packaging a single, universal native binary via Electron compilation tools like PyInstaller or Nuitka 28 is highly problematic and likely to fail across heterogeneous fleets. The framework's reliance on disparate hardware-level containerization strategies validates the continued use of Docker as the execution vehicle. Consequently, the Electron application must not *contain* the training environment; rather, it must act as an *orchestrator* that controls a localized, pre-installed Docker daemon.29

### **4.3 Architectural Optimization: Docker Socket Interaction via Node.js**

Rather than bloating the Electron build process, the desktop application should leverage the dockerode Node.js library—or standard Node child\_process execution modules—within the Main Process to manage the federated learning Docker containers dynamically.29 The Electron Renderer UI can provide an intuitive interface for gathering configuration parameters, such as allowing the user to select the target hardware profile (e.g., Discrete GPU versus Jetson SoC). These parameters are passed securely over the IPC bridge. The Main Process then programmatically constructs the appropriate Docker command—injecting the necessary \--device mounts for Jetson or \--gpus all for discrete workstations—and streams the resulting Docker container logs back to the user interface in real-time.

**Example Node.js Orchestration Logic (Conceptual Main Process)**

JavaScript

const Docker \= require('dockerode');  
const docker \= new Docker({ socketPath: '/var/run/docker.sock' });

async function startFederatedClient(hardwareProfile, projectId) {  
    let hostConfig \= {  
        Binds: \['/local/model/path:/app/model'\]  
    };

    // Dynamically inject hardware-specific Docker configurations  
    if (hardwareProfile \=== 'discrete') {  
        hostConfig.DeviceRequests \=\]  
        }\];  
    } else if (hardwareProfile \=== 'jetson') {  
        hostConfig.Devices \= \[  
            { PathOnHost: '/dev/nvhost-ctrl', PathInContainer: '/dev/nvhost-ctrl', CgroupPermissions: 'rwm' },  
            { PathOnHost: '/dev/nvhost-ctrl-gpu', PathInContainer: '/dev/nvhost-ctrl-gpu', CgroupPermissions: 'rwm' }  
        \];  
    }

    const container \= await docker.createContainer({  
        Image: 'fedlearn-client:latest',  
        Cmd: \['python', 'client.py', '--project-id', projectId\],  
        HostConfig: hostConfig  
    });

    await container.start();  
}

## **5\. Security Posture: Electron Vulnerabilities and Defense-in-Depth**

### **5.1 Conceptual Foundation: Context Isolation and The Docker Socket Vector**

Web applications operate under a strict zero-trust model, relying on the robust sandboxing capabilities of modern browsers to prevent malicious scripts from accessing the underlying operating system. Electron, however, merges these web technologies with system-level access. If a malicious actor successfully executes a Cross-Site Scripting (XSS) payload within an unsecured Electron Renderer process, that script can theoretically leverage the Node.js runtime to execute arbitrary code on the host machine—a scenario known as Remote Code Execution (RCE).23

Compounding this inherent risk is the proposed integration with Docker. The Docker daemon socket (/var/run/docker.sock) is a Unix domain socket that grants root-equivalent permissions over the host system.33 If an Electron application directly mounts or manages this socket to orchestrate the federated learning clients, and the Renderer application is subsequently compromised via XSS, the attacker gains full, unrestricted control over the host's container infrastructure, allowing them to spawn malicious cryptomining containers or mount the root filesystem.

### **5.2 Codebase Analysis: Context Isolation and Information Leakage**

The deployment proposal explicitly indicates a critical need to standardize authentication via jwtToken and systematically remove exposed console.log statements during the transition to the desktop.35 Standard console.log statements that remain in production Electron builds pose a significant security threat; they can inadvertently leak highly sensitive initialization parameters, cryptographic jwtToken strings, or internal file paths directly to the Chromium DevTools console. This information leakage provides attackers with the necessary architectural context to craft highly targeted payloads.36

Furthermore, legacy Electron architectures often relied on the remote module or completely disabled contextIsolation to simplify communication between the Renderer UI and the backend Main processes. The remote module has been formally deprecated due to severe security implications—it allowed the Renderer process to invoke Main process methods directly, entirely bypassing security boundaries and enabling prototype pollution attacks.24

### **5.3 Architectural Optimization: Webpack AST Transformations and ContextBridge**

To harden the desktop client against these multifaceted vectors, two primary defensive strategies must be structurally enforced across the build pipeline:

1. **Strict Context Isolation (contextBridge):** The Electron application must be instantiated with nodeIntegration: false and contextIsolation: true.24 All communication regarding Docker orchestration, file system access, or authentication token management must flow exclusively through heavily validated IPC channels. These channels are exposed to the UI via a secure preload.js script utilizing the contextBridge.exposeInMainWorld API.37  
2. **Automated Console Stripping via Webpack:** Instead of relying on developers to manually locate and delete console.log statements—an error-prone methodology—the build pipeline should automatically strip all debugging output during the packaging phase. The Webpack TerserPlugin handles this efficiently by executing an Abstract Syntax Tree (AST) modification during JavaScript minification, entirely excising the logging function calls from the compiled binary.38

### **5.4 Code Refactoring: Hardening the Electron Build and IPC Security**

The following refactoring provides the precise configurations required to implement both Context Isolation and automated log stripping, ensuring a robust defense-in-depth posture.

**Securing the Build: Webpack TerserPlugin Configuration (webpack.prod.config.js)**

JavaScript

const TerserPlugin \= require('terser-webpack-plugin');

module.exports \= {  
  mode: 'production',  
  optimization: {  
    minimize: true,  
    minimizer:,  
  },  
};

**Securing the Bridge: Validated IPC Communication (preload.js)**

JavaScript

const { contextBridge, ipcRenderer } \= require('electron');

// Secure contextBridge implementation: Only exposes strictly defined,   
// parameter-validated functions to the global window object.  
// Absolutely prevents exposing the entire IPC or Node.js runtime to the Renderer process.  
contextBridge.exposeInMainWorld('fedLearnAPI', {  
      
    // The UI requests the Main process to start the training Docker container.  
    // The input parameter is strictly validated before transmission over the IPC bridge.  
    startTraining: (hardwareProfile) \=\> {  
        const allowedProfiles \= \['discrete', 'jetson', 'cpu'\];  
        if (allowedProfiles.includes(hardwareProfile)) {  
            ipcRenderer.send('docker-start-training', hardwareProfile);  
        } else {  
            console.error("Security Violation: Invalid hardware profile requested.");  
        }  
    },  
      
    // The Main process streams asynchronous training logs back to the UI.  
    onTrainingLog: (callback) \=\> {  
        ipcRenderer.on('training-log', (\_event, value) \=\> callback(value));  
    }  
});

## **6\. Risk and Mitigation Matrix**

The following table synthesizes the primary architectural mismatches, technical risks, and security gaps identified during the system analysis, alongside the concrete mitigations necessary for successful enterprise deployment.

| Technical Risk / Gap | Component Impacted | Impact Severity | Recommended Mitigation Strategy |
| :---- | :---- | :---- | :---- |
| **Ephemeral Port Exhaustion & Isolation Failure** | AWS Migration (FlowerServerManager) | **Critical** | Deprecate ProcessBuilder. Utilize AWS SDK RunTask API (Fargate) to launch isolated fl\_server.py containers. Use ECS Service Connect for dynamic DNS service discovery. |
| **Silent Connection Drops (WAN Latency)** | gRPC Communication (grpc\_client.py) | **High** | Implement native gRPC HTTP/2 Keepalive channel arguments (grpc.keepalive\_time\_ms, permit\_without\_calls) configured strictly below AWS ALB/NLB idle timeout limits (e.g., \< 350s for NLB). |
| **PyTorch OOM on Tensor Serialization** | gRPC Communication (\_submit\_update\_stream) | **High** | Refactor streaming logic to utilize Python generator functions (yield) and memoryview. Stream byte chunks directly from the in-memory buffer without creating duplicate intermediate byte arrays. |
| **Massive Electron Package Bloat (\>4GB)** | Electron Desktop Build | **Medium** | Do not bundle the Python interpreter, PyTorch, CUDA, or Docker inside the app.asar. Design the Electron Main Process to act solely as a system orchestrator that invokes commands to an already installed host Docker daemon. |
| **Hardware Driver Disparities** | Docker Runtime (Jetson vs. Discrete GPUs) | **High** | Implement conditional execution logic in the Electron Main process. Inject \--gpus all for standard workstations and strict \--device mappings (/dev/nvhost-ctrl) for NVIDIA Jetson targets to prevent kernel-level runtime hangs. |
| **Remote Code Execution (RCE) via XSS** | Electron Renderer Process | **Critical** | Enforce nodeIntegration: false and contextIsolation: true. Route all Docker daemon socket controls strictly through type-validated contextBridge IPC channels configured within preload.js. |
| **Information Leakage via DevTools** | Electron Production Release | **Medium** | Implement Webpack's TerserPlugin with the drop\_console: true directive in the production build pipeline to programmatically strip all debugging and authentication token output from the packaged codebase. |

## **7\. Recommended AWS Networking Architecture**

To support the migration from a monolithic Spring Boot paradigm to a highly available, distributed cloud architecture, the AWS infrastructure must be strategically partitioned to ensure high performance, security, and scalability. The following outline delineates the optimal conceptual networking topology required to route Edge client traffic across the WAN to the dynamically spawned Python federated servers securely and reliably.

The architecture is structurally divided into three distinct operational tiers within a primary Virtual Private Cloud (VPC), distributed across multiple Availability Zones (AZs) for fault tolerance:

**1\. Public Boundary Layer (Ingress Routing & Load Balancing):**

The boundary layer is responsible for terminating internet-facing connections and routing them securely into the private subnets.

* **AWS Internet Gateway (IGW):** Handles all ingress traffic originating from distributed Electron edge clients traversing the public internet.  
* **Network Load Balancer (NLB):** Deployed within Public Subnets. An NLB operates at Layer 4 (Transport Layer) and is highly recommended over an Application Load Balancer (ALB) for gRPC architectures requiring persistent, long-lived bidirectional streams. The NLB provides an extended default idle timeout (350 seconds compared to the ALB's 60 seconds) and introduces significantly lower latency for raw TCP/HTTP2 traffic routing.  
* **Public Security Group:** A restrictive firewall policy allowing ingress traffic exclusively on port 443, ensuring all gRPC and WebSocket traffic is strictly TLS encrypted before entering the VPC.

**2\. Application Tier (Compute & Orchestration in Private Subnets):**

The compute layer is isolated from the internet, accessible only via the NLB. It utilizes AWS Fargate for serverless container execution, eliminating the need to manage underlying EC2 instances.

* **Spring Boot Orchestration Cluster (ECS Fargate):** The central API logic resides here. When a user requests a new training round, the Spring Boot container utilizes its attached IAM Task Role to securely invoke the ECS RunTask API to spawn the Python servers.  
* **Federated Learning Python Servers (ECS Fargate):** Dynamically provisioned by the Spring Boot API. Operating in awsvpc network mode, each server task is assigned a unique, ephemeral private IP address and a dedicated Elastic Network Interface (ENI). They do not rely on static port bindings or host-level port conflicts.  
* **AWS Cloud Map & ECS Service Connect:** Instead of routing traffic from the NLB directly to random IPs—which causes severe issues during scaling events—the NLB targets an Envoy proxy layer managed by ECS Service Connect. As Python tasks spin up, they register their ephemeral private IPs and dynamically assigned ports into a highly available private Cloud Map namespace (e.g., fl-server.internal). The Envoy sidecars intercept the incoming gRPC traffic from the NLB and route it transparently to the correct downstream Python container, enabling near-instantaneous service discovery.

**3\. Data Tier (State Management in Isolated Subnets):**

The deepest layer of the VPC is reserved for persistent state and metadata storage.

* **Amazon RDS (PostgreSQL):** Hosts the relational database containing project metadata, client registry states, and system configurations. Deployed across Multiple Availability Zones (Multi-AZ) to guarantee automatic failover and redundancy.  
* **Database Security Group:** Ingress is explicitly restricted to only accept connections originating from the Application Tier Security Group assigned to the Spring Boot cluster. Direct access from the public internet or the Python servers is structurally prohibited, enforcing the principle of least privilege.

#### **Works cited**

1. PortMapping \- Amazon Elastic Container Service \- AWS Documentation, accessed April 14, 2026, [https://docs.aws.amazon.com/AmazonECS/latest/APIReference/API\_PortMapping.html](https://docs.aws.amazon.com/AmazonECS/latest/APIReference/API_PortMapping.html)  
2. Use an Application Load Balancer for Amazon ECS \- Amazon Elastic Container Service, accessed April 14, 2026, [https://docs.aws.amazon.com/AmazonECS/latest/developerguide/alb.html](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/alb.html)  
3. FedLearn: Distributed Federated Learning Framework  
4. Use service discovery to connect Amazon ECS services with DNS names, accessed April 14, 2026, [https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-discovery.html](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-discovery.html)  
5. Use a Network Load Balancer for Amazon ECS \- Amazon Elastic Container Service, accessed April 14, 2026, [https://docs.aws.amazon.com/AmazonECS/latest/developerguide/nlb.html](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/nlb.html)  
6. Dynamic Port Mapping for Amazon ECS \- AWS, accessed April 14, 2026, [https://aws.amazon.com/video/watch/f8f423efeb8/](https://aws.amazon.com/video/watch/f8f423efeb8/)  
7. run-task — AWS CLI 2.34.29 Command Reference, accessed April 14, 2026, [https://docs.aws.amazon.com/cli/latest/reference/ecs/run-task.html](https://docs.aws.amazon.com/cli/latest/reference/ecs/run-task.html)  
8. Best practices for connecting Amazon ECS services in a VPC, accessed April 14, 2026, [https://docs.aws.amazon.com/AmazonECS/latest/developerguide/networking-connecting-services.html](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/networking-connecting-services.html)  
9. Amazon ECS Service Connect components, accessed April 14, 2026, [https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-connect-concepts-deploy.html](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-connect-concepts-deploy.html)  
10. Interconnect Amazon ECS services \- AWS Documentation, accessed April 14, 2026, [https://docs.aws.amazon.com/AmazonECS/latest/developerguide/interconnecting-services.html](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/interconnecting-services.html)  
11. New – Amazon ECS Service Connect Enabling Easy Communication Between Microservices | AWS News Blog, accessed April 14, 2026, [https://aws.amazon.com/blogs/aws/new-amazon-ecs-service-connect-enabling-easy-communication-between-microservices/](https://aws.amazon.com/blogs/aws/new-amazon-ecs-service-connect-enabling-easy-communication-between-microservices/)  
12. Performance best practices with gRPC \- Microsoft Learn, accessed April 14, 2026, [https://learn.microsoft.com/en-us/aspnet/core/grpc/performance?view=aspnetcore-10.0](https://learn.microsoft.com/en-us/aspnet/core/grpc/performance?view=aspnetcore-10.0)  
13. How to Implement gRPC Keepalive for Long-Lived Connections \- OneUptime, accessed April 14, 2026, [https://oneuptime.com/blog/post/2026-01-08-grpc-keepalive-connections/view](https://oneuptime.com/blog/post/2026-01-08-grpc-keepalive-connections/view)  
14. Node gRPC Keepalive That Actually Works | by Duckweave \- Medium, accessed April 14, 2026, [https://medium.com/@duckweave/node-grpc-keepalive-that-actually-works-98ef2ccc6390](https://medium.com/@duckweave/node-grpc-keepalive-that-actually-works-98ef2ccc6390)  
15. Performance Best Practices | gRPC, accessed April 14, 2026, [https://grpc.io/docs/guides/performance/](https://grpc.io/docs/guides/performance/)  
16. Keepalive \- gRPC, accessed April 14, 2026, [https://grpc.io/docs/guides/keepalive/](https://grpc.io/docs/guides/keepalive/)  
17. TorchServe gRPC API — PyTorch/Serve master documentation, accessed April 14, 2026, [https://docs.pytorch.org/serve/grpc\_api.html](https://docs.pytorch.org/serve/grpc_api.html)  
18. Chunk transfer vs grpc streaming \- Stack Overflow, accessed April 14, 2026, [https://stackoverflow.com/questions/71070209/chunk-transfer-vs-grpc-streaming](https://stackoverflow.com/questions/71070209/chunk-transfer-vs-grpc-streaming)  
19. Serialization overhead of multiprocessing \- PyTorch Forums, accessed April 14, 2026, [https://discuss.pytorch.org/t/serialization-overhead-of-multiprocessing/29628](https://discuss.pytorch.org/t/serialization-overhead-of-multiprocessing/29628)  
20. Deploying Machine Learning Models with PyTorch, gRPC and asyncio \- GitHub, accessed April 14, 2026, [https://github.com/roboflow/deploy-models-with-grpc-pytorch-asyncio](https://github.com/roboflow/deploy-models-with-grpc-pytorch-asyncio)  
21. Serialization — Ray 2.54.1 \- Ray Docs, accessed April 14, 2026, [https://docs.ray.io/en/latest/ray-core/objects/serialization.html](https://docs.ray.io/en/latest/ray-core/objects/serialization.html)  
22. How to use zero-copy serialization libraries without moving the data? \- Stack Overflow, accessed April 14, 2026, [https://stackoverflow.com/questions/48092129/how-to-use-zero-copy-serialization-libraries-without-moving-the-data](https://stackoverflow.com/questions/48092129/how-to-use-zero-copy-serialization-libraries-without-moving-the-data)  
23. Electron App Security Risks and CVE Case Studies \- SecureLayer7, accessed April 14, 2026, [https://blog.securelayer7.net/electron-app-security-risks/](https://blog.securelayer7.net/electron-app-security-risks/)  
24. Context Isolation \- Electron, accessed April 14, 2026, [https://electronjs.org/docs/latest/tutorial/context-isolation](https://electronjs.org/docs/latest/tutorial/context-isolation)  
25. Get Started \- PyTorch, accessed April 14, 2026, [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)  
26. NVIDIA Container Runtime, accessed April 14, 2026, [https://developer.nvidia.com/container-runtime](https://developer.nvidia.com/container-runtime)  
27. Nvidia/cuda doesn't work on Docker Desktop but works on Docker Engine, accessed April 14, 2026, [https://forums.docker.com/t/nvidia-cuda-doesnt-work-on-docker-desktop-but-works-on-docker-engine/130668](https://forums.docker.com/t/nvidia-cuda-doesnt-work-on-docker-desktop-but-works-on-docker-engine/130668)  
28. \[D\] Best way to package Pytorch models as a standalone application \- Reddit, accessed April 14, 2026, [https://www.reddit.com/r/MachineLearning/comments/1050cw1/d\_best\_way\_to\_package\_pytorch\_models\_as\_a/](https://www.reddit.com/r/MachineLearning/comments/1050cw1/d_best_way_to_package_pytorch_models_as_a/)  
29. Docker-Containerized Automation App with Node, Python & Electron \- Adevait, accessed April 14, 2026, [https://adevait.com/nodejs/docker-container-automation-node-python-electron](https://adevait.com/nodejs/docker-container-automation-node-python-electron)  
30. How to run an electron app on docker \- Stack Overflow, accessed April 14, 2026, [https://stackoverflow.com/questions/39930223/how-to-run-an-electron-app-on-docker](https://stackoverflow.com/questions/39930223/how-to-run-an-electron-app-on-docker)  
31. How to run Electron on Linux on Docker on Mac \- Jake Donham, accessed April 14, 2026, [https://jaked.org/blog/2021-02-18-How-to-run-Electron-on-Linux-on-Docker-on-Mac](https://jaked.org/blog/2021-02-18-How-to-run-Electron-on-Linux-on-Docker-on-Mac)  
32. Electronjs CVEs and Security Vulnerabilities \- OpenCVE, accessed April 14, 2026, [https://app.opencve.io/cve/?vendor=electronjs](https://app.opencve.io/cve/?vendor=electronjs)  
33. Docker Security \- OWASP Cheat Sheet Series, accessed April 14, 2026, [https://cheatsheetseries.owasp.org/cheatsheets/Docker\_Security\_Cheat\_Sheet.html](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)  
34. How secure is Docker? \- Reddit, accessed April 14, 2026, [https://www.reddit.com/r/docker/comments/p2yamw/how\_secure\_is\_docker/](https://www.reddit.com/r/docker/comments/p2yamw/how_secure_is_docker/)  
35. Using console.log() in Electron app for debugging \- Prospera Soft, accessed April 14, 2026, [https://prosperasoft.com/blog/full-stack/frontend/electronjs/electron-console-log/](https://prosperasoft.com/blog/full-stack/frontend/electronjs/electron-console-log/)  
36. How to Remove Console Statements From Production Build in 3 Easy Ways \- HackerNoon, accessed April 14, 2026, [https://hackernoon.com/how-to-remove-console-statements-from-production-build-in-3-easy-ways](https://hackernoon.com/how-to-remove-console-statements-from-production-build-in-3-easy-ways)  
37. Build and Secure an Electron App \- OpenID, OAuth, Node.js, and Express \- Auth0, accessed April 14, 2026, [https://auth0.com/blog/securing-electron-applications-with-openid-connect-and-oauth-2/](https://auth0.com/blog/securing-electron-applications-with-openid-connect-and-oauth-2/)  
38. Remove console.log with TerserWebpackPlugin \- Stack Overflow, accessed April 14, 2026, [https://stackoverflow.com/questions/54561070/remove-console-log-with-terserwebpackplugin](https://stackoverflow.com/questions/54561070/remove-console-log-with-terserwebpackplugin)  
39. Remove console statements: Webpack 5 | by Shaan \- Medium, accessed April 14, 2026, [https://medium.com/@shaangontia/remove-console-statements-webpack-5-38455e4b471a](https://medium.com/@shaangontia/remove-console-statements-webpack-5-38455e4b471a)