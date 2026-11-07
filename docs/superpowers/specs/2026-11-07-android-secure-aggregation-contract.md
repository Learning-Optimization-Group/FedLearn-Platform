# Android Secure Aggregation Contract: Cross-Language Specification

## 1. Overview & Scope

This specification formalizes the cross-language secure aggregation protocol (Area 6 of the MSc Capstone reliability program) between the Python federated backend/simulation framework and mobile clients (Android Java/C++ and React Native).

The implementation builds on **LightSecAgg** (So, Güler, Avestimehr, 2021), designed specifically for dropout-resilient, communication-efficient secure aggregation over zeroth-order gradient scalars and first-order linear gradient vectors.

---

## 2. Threat Model & Security Properties

1. **Adversary Model**:
   - **Honest-but-Curious Server**: The server follows the protocol execution honestly but attempts to inspect or infer individual client model updates or gradient vectors from wire transcripts.
   - **Colluding Participants**: Up to $T - 1$ clients may collude with each other and/or the server to reveal the private vector of an honest client.
   - **Unreliable Network / Dropouts**: Clients may drop out at any stage of the round (e.g. Wi-Fi disconnection, process suspension, battery saver). The protocol must maintain exact unmasked aggregation of surviving participants without per-dropout mask reconstruction overhead.

2. **Security Guarantees**:
   - **Information-Theoretic Privacy**: Provided at least $T$ honest clients participate and at least $U \ge T$ clients survive, neither the server nor any coalition of $< T$ colluding clients learns any information about individual updates beyond the aggregate $\sum_{i \in \mathcal{S}} \mathbf{x}_i$.
   - **One-Shot Aggregate Recovery**: Unlike classic SecAgg (Bonawitz et al., 2017) which requires reconstructing individual masks for every dropped client ($O(|\mathcal{D}|)$ Lagrange decodes), LightSecAgg requires exactly **one** single interpolation at the server over the summed shares of survivors, regardless of how many participants drop out.

---

## 3. Mathematical & Cryptographic Foundation

### 3.1 Finite Field Definition
Computations are performed over the finite field $\mathbb{F}_p$ where $p$ is the Mersenne prime:
$$p = 2^{31} - 1 = 2,147,483,647$$
The choice of $M_{31}$ ensures:
1. Exact arithmetic on standard 32-bit and 64-bit integer registers without floating-point rounding errors.
2. Fast modular reduction: $a \pmod{2^{31}-1} = (a \ \& \ (2^{31}-1)) + (a \gg 31)$.
3. Inversion via Fermat's Little Theorem: $a^{-1} \equiv a^{p-2} \pmod p$.

### 3.2 Fixed-Point Quantization
Continuous floating-point gradient values $x \in \mathbb{R}$ are quantized to $\mathbb{F}_p$ using a uniform scaling factor $S = 1,000,000$:
$$q(x) = \left\lfloor x \cdot S + 0.5 \right\rfloor \pmod p$$
De-quantization at the server after unmasking:
$$\hat{x} = \frac{(y \pmod p) - p \cdot \mathbb{I}_{y > p/2}}{S}$$

### 3.3 Shamir Secret Sharing
For client $i$, its local random mask vector $\mathbf{z}_i \in \mathbb{F}_p^d$ is generated via a cryptographically secure pseudo-random generator keyed by $\text{HMAC-SHA256}(K_i, \text{round\_seed})$.
To share $\mathbf{z}_i$ among $N$ clients with privacy threshold $T$:
1. Dealer $i$ samples $T - 1$ random coefficient vectors $\mathbf{a}_{i,1}, \dots, \mathbf{a}_{i,T-1} \sim \mathcal{U}(\mathbb{F}_p^d)$.
2. The polynomial is defined as:
   $$f_i(x) = \mathbf{z}_i + \sum_{k=1}^{T-1} \mathbf{a}_{i,k} x^k \pmod p$$
3. The share for client $j \in \{1, \dots, N\}$ is evaluated at $x = j$:
   $$s_{i \to j} = f_i(j) \pmod p$$

---

## 4. Multi-Stage Protocol Execution Flow

```mermaid
sequenceDiagram
    autonumber
    actor C_i as Client i
    actor C_j as Client j (Cohort)
    participant S as Federated Server

    Note over C_i,S: Stage 1: Cohort Discovery & Share Distribution
    S->>C_i: Advertise Round(cohort_size N, threshold T, round_seed)
    C_i->>C_i: Sample mask z_i, compute Shamir shares s_{i -> j}
    C_i->>S: SubmitShares(target_client_id=j, encrypted_share)
    S->>C_j: Route assigned shares from all dealers

    Note over C_i,S: Stage 2: Masked Upload
    C_i->>C_i: Train local model, quantize gradients q(x_i)
    C_i->>C_i: Compute masked vector y_i = (q(x_i) + z_i) mod p
    C_i->>S: SubmitMaskedUpdate(y_i, client_id=i)

    Note over S: Stage 3: Dropout Identification
    S->>S: Deadline expires; identify surviving cohort S (|S| >= T)
    S->>C_j: Broadcast surviving client list S

    Note over C_j,S: Stage 4: Aggregate Share Return & Unmasking
    C_j->>C_j: Compute summed share S_j = sum_{i in S} s_{i -> j} mod p
    C_j->>S: SubmitAggregateShare(S_j, client_id=j)
    S->>S: Lagrange interpolate at x=0 to recover Z_S = sum_{i in S} z_i mod p
    S->>S: Unmask aggregate: sum_{i in S} q(x_i) = (sum_{i in S} y_i - Z_S) mod p
```

### Protocol Steps:
1. **Stage 1 (Setup & Share Distribution)**:
   - Server announces round parameters: active cohort $\mathcal{C}$ with size $N$, threshold $T$, and round seed.
   - Each client $i$ generates mask $\mathbf{z}_i$, generates $N$ Shamir shares, and transmits them to the server for pairwise routing to target holders.
2. **Stage 2 (Masked Upload)**:
   - Each client $i$ computes its gradient update, quantizes it, adds mask $\mathbf{z}_i \pmod p$, and sends $\mathbf{y}_i = q(\mathbf{x}_i) + \mathbf{z}_i \pmod p$.
3. **Stage 3 (Dropout Fencing)**:
   - Server waits for round deadline. If $|S| < T$, the round aborts.
   - If $|S| \ge T$, server freezes the surviving set $S$ and notifies all surviving clients.
4. **Stage 4 (Aggregate Share Return)**:
   - Each surviving client $j \in S$ sums only the shares it holds from surviving dealers:
     $$S_j = \sum_{i \in S} s_{i \to j} \pmod p$$
   - Each client $j$ sends this single aggregated share $S_j$ to the server.
5. **Stage 5 (Single Lagrange Decode & Unmasking)**:
   - Due to polynomial linearity:
     $$\sum_{i \in S} f_i(x) = \left(\sum_{i \in S} \mathbf{z}_i\right) + \sum_{k=1}^{T-1} \left(\sum_{i \in S} \mathbf{a}_{i,k}\right) x^k \pmod p$$
   - The server interpolates $T$ received shares $\{S_j\}_{j \in S_{sub}}$ at $x = 0$ using Lagrange coefficients:
     $$\mathbf{Z}_S = \sum_{j \in S_{sub}} \lambda_j \cdot S_j \pmod p = \sum_{i \in S} \mathbf{z}_i \pmod p$$
     where $\lambda_j = \prod_{k \in S_{sub}, k \ne j} \frac{-k}{j - k} \pmod p$.
   - The server subtracts $\mathbf{Z}_S$ from the sum of masked vectors:
     $$\sum_{i \in S} q(\mathbf{x}_i) = \left(\sum_{i \in S} \mathbf{y}_i - \mathbf{Z}_S\right) \pmod p$$

---

## 5. Cross-Language Data Layout & Protocol Buffers

### 5.1 Protobuf Contract (`proto/fedlearn/v2/fedlearn.proto`)
The gRPC wire messages adhere to the following layout:

```protobuf
message MaskedVectorPayload {
  int32 vector_length = 1;
  repeated int64 elements = 2; // Elements in field [0, 2^31 - 2]
  int64 scale = 3;             // Default 1,000,000
}

message SubmitSharesRequest {
  string run_id = 1;
  int32 round_number = 2;
  string dealer_client_id = 3;
  map<string, bytes> encrypted_shares = 4; // recipient_id -> sealed share
}

message SubmitAggregateShareRequest {
  string run_id = 1;
  int32 round_number = 2;
  string holder_client_id = 3;
  repeated int64 aggregate_share_elements = 4;
}
```

### 5.2 Endianness & Memory Layout
- **Endianness**: Fixed Little-Endian for all multi-byte integers.
- **Quantized Coefficients**: Represented as contiguous arrays of `int64` (to avoid overflow during accumulation prior to modulo operations).
- **C++ Native Core Binding**: Mobile implementation (`mobile_client/android/app/src/main/cpp`) interacts with `std::vector<int64_t>` via JNI, matching Python's `torch.int64` tensor layout bit-for-bit.

---

## 6. Client Capability Guarding & Refusal Logic

Because weight-update federated learning involves high-dimensional non-linear tensors (e.g., convolution kernels, attention projections, BatchNorm parameters) where secure aggregation overhead and clipping assumptions differ from scalar gradient descent:
1. **Scope Boundary**: LightSecAgg is strictly scoped to gradient scalars and linear gradient vectors (e.g. `DeComFL` and linear classification heads).
2. **Client-Side Guard**: The mobile client enforces explicit capability checking in `mobile_client/src/lib/runJoin.ts`:
   - If an execution contract specifies `security.secureAggregation == SECAGG_LIGHTSECAGG_SCALAR`, physical Android devices verify native arithmetic support before joining.
   - If a contract attempts to specify secure aggregation over arbitrary dense weight updates (`ARM_FULL`), the client rejects the contract with `MobileSecureAggregationUnsupportedError`, preventing silent protocol corruption.
