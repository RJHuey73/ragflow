# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAGFlow is an open-source RAG (Retrieval-Augmented Generation) engine based on deep document understanding. It's a full-stack application with:

- Python backend (Quart-based async API server — Quart is the async reimplementation of Flask)
- React/TypeScript frontend (built with vitejs)
- Background task executor workers (separate Python processes, Redis-queue-driven)
- Peewee ORM for database models (not SQLAlchemy)
- Multiple data stores (MySQL/PostgreSQL, Elasticsearch/Infinity/OpenSearch/OceanBase, Redis, MinIO)

PLACEHOLDER