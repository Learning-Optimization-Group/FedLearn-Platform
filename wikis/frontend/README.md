# FedLearn Frontend Documentation

Welcome to the FedLearn Frontend Wiki! This directory contains comprehensive documentation on the architecture, technical decisions, components, and implementation details of the React-based client application.

## Table of Contents

1. [Architecture & State Management](./Architecture.md)
   - Tech Stack Overview (resolved versions, not ranges)
   - Project Structure
   - Global State, Contexts and Service Stores
   - Vite modes ↔ Spring profiles, the dev proxy, `strictPort`
   - Build guards, linting and the Vitest suite
2. [Authentication & Routing](./Routing_and_Auth.md)
   - React Router Configuration & the full route map
   - `ProtectedRoute` / `RoleRoute` guards
   - HttpOnly Cookie Auth Implementation
   - Identity refresh on focus / visibility
3. [API & Services](./API_and_Services.md)
   - Axios Configuration & Interceptors
   - The typed service layer (`apiServices.ts`, `artifactService.ts`)
   - STOMP-over-WebSocket destinations & honest connection state
   - Log Store & Real-time Telemetry
4. [UI & Components](./UI_and_Components.md)
   - The **Ledger** design system & the generated token pipeline
   - Component conventions and the `ui/` primitives
   - The project-creation picker and the **training arm** trade-off panel

## Overview

The FedLearn frontend is a modern Single Page Application (SPA) built with React 19, TypeScript, and Vite. It serves as the primary control plane for the federated learning system, allowing users to:

- Authenticate and manage access securely using HttpOnly cookies — no token is ever readable from JavaScript.
- View and manage projects, models, datasets, the artifact registry and the adapter marketplace.
- Choose *how* a project trains: the model recipe, the optimizer, and — where the recipe offers more than one — the **training arm**, shown with the measured cost of the choice (see [UI & Components](./UI_and_Components.md#the-project-creation-picker-and-the-training-arm)).
- Monitor live training across multiple clients over STOMP-over-WebSocket.
- Run inference against a trained model, and (as a platform admin) explore audit events and benchmark runs.
- Use the current **Ledger** design system (navy structural ink `#1C314D` on quiet paper — `#F6F3EE` canvas, white cards — with a single Hanken Grotesk type family and JetBrains Mono for logs/ids), which superseded the earlier **Ember** system (warm paper + burnt orange + Bricolage Grotesque display, with an AMOLED-black dark family) and the "Instrument" tokens before it. See [UI & Components](./UI_and_Components.md) for the token pipeline.

## Quick reference

| Thing | Value |
|---|---|
| Package | `fedlearn-frontend`, version `1.4.1-beta` (`frontend/package.json`) |
| Dev server | `npm run dev` → `http://localhost:5173` (`strictPort: true` — it will not shift to 5174) |
| Backend it expects | Spring Boot on `http://localhost:8081`, API base `…/api` |
| Node | 24 (repo-pinned in `.nvmrc` / `.tool-versions`; CI uses `node-version: '24'`) |
| Test runner | **Vitest** + Testing Library + jsdom (`npm run test:run`, `npm run test:coverage`) |
| CI gate | `npm run lint` → `npx tsc --noEmit` → `npm run test:coverage` → `npm run build` (`.github/workflows/ci.yml`, path-filtered on `frontend/**`) |

Start by exploring the [Architecture](./Architecture.md) guide to understand the fundamental building blocks of the application.
