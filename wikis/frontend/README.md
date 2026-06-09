# FedLearn Frontend Documentation

Welcome to the FedLearn Frontend Wiki! This directory contains comprehensive documentation on the architecture, technical decisions, components, and implementation details of the React-based client application.

## Table of Contents

1. [Architecture & State Management](./Architecture.md)
   - Tech Stack Overview
   - Project Structure
   - Global State and Contexts
2. [Authentication & Routing](./Routing_and_Auth.md)
   - React Router Configuration
   - Protected Routes & Auth Guard
   - HttpOnly Cookie Auth Implementation
3. [API & Services](./API_and_Services.md)
   - Axios Configuration & Interceptors
   - WebSocket Integration
   - Log Store & Real-time Telemetry
4. [UI & Components](./UI_and_Components.md)
   - Design System & Tailwind CSS
   - Legacy vs. V2 (Redesign) Components
   - Reusable Component Patterns

## Overview

The FedLearn frontend is a modern Single Page Application (SPA) built with React 19, TypeScript, and Vite. It serves as the primary control plane for the federated learning system, allowing users to:
- Authenticate and manage access securely using HttpOnly cookies.
- View and manage projects, models, and federated learning datasets.
- Monitor live training sessions across multiple clients via WebSockets.
- Seamlessly transition between a legacy UI and a redesigned (Apple-inspired) V2 interface.

Start by exploring the [Architecture](./Architecture.md) guide to understand the fundamental building blocks of the application.
