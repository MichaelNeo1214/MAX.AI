# 🤖 MAX.AI

> **A personal AI assistant built by Michael.**

MAX.AI is a personal AI assistant project focused on building a fast, private, and customizable conversational AI experience.

The project is currently under **active development**. MAX.AI is still experimental, and some features may not work as expected.

---

## ✨ About MAX.AI

**MAX.AI** is an AI assistant designed to become a personal, customizable AI system that can run locally and evolve over time.

The goal isn't simply to create another chatbot, but to build an AI assistant that can be:

* 🧠 Intelligent and conversational
* ⚡ Fast and lightweight
* 🔒 Privacy-focused
* 💻 Capable of running locally
* 🛠️ Highly customizable
* 🚀 Continuously improved

MAX.AI is currently powered by a **Qwen3.5 9B quantized model** and is being optimized for local hardware.

---

## 🧠 Current Model

MAX.AI currently uses:

**Qwen3.5-9B**

with the:

**Qwen3.5-9B-Q4_K_M** quantization.

The model is currently being integrated and optimized as part of the MAX.AI runtime.

> ⚠️ **Development Notice:**
> The model integration is still being worked on. Expect bugs, instability, unexpected responses, and incomplete functionality.

---

## 🏗️ Architecture

The current MAX.AI stack is roughly structured like this:

```text
                    ┌─────────────────┐
                    │     MAX.AI      │
                    │   AI Assistant  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Chat Interface │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  AI Runtime     │
                    │   / Inference   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Qwen3.5 9B      │
                    │    Q4_K_M       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Local Hardware  │
                    └─────────────────┘
```

The architecture is expected to evolve as development continues.

---

## 🚧 Development Status

### Current Status

**🟡 In Development**

MAX.AI is currently functional in parts, but it is **not considered production-ready**.

Some known issues include:

* 🐛 Model integration bugs
* 🐛 Inference instability
* 🐛 Unexpected model responses
* 🐛 Performance optimization still in progress
* 🐛 UI/UX improvements
* 🐛 Context handling issues
* 🐛 Memory management improvements
* 🐛 Various experimental features

Breaking changes may happen frequently during development.

---

## 🗺️ Roadmap

### Phase 1 — Foundation

* [x] Initial MAX.AI project
* [x] Local LLM experimentation
* [x] Qwen model integration
* [x] Q4_K_M quantization testing
* [ ] Stable inference pipeline
* [ ] Basic conversation system

### Phase 2 — Core AI

* [ ] Better context management
* [ ] Conversation memory
* [ ] System prompt management
* [ ] Streaming responses
* [ ] Improved inference performance
* [ ] Error handling
* [ ] Model configuration

### Phase 3 — MAX.AI Interface

* [ ] Modern chat UI
* [ ] Responsive design
* [ ] Conversation history
* [ ] Markdown rendering
* [ ] Code syntax highlighting
* [ ] File interaction
* [ ] Settings panel

### Phase 4 — Intelligence

* [ ] Long-term memory
* [ ] Retrieval-Augmented Generation (RAG)
* [ ] Document understanding
* [ ] Tool calling
* [ ] Web access
* [ ] Personal knowledge base

### Phase 5 — MAX.AI Ecosystem

* [ ] Plugin system
* [ ] Custom tools
* [ ] API
* [ ] Multiple model support
* [ ] Local & remote inference
* [ ] Voice interaction
* [ ] Vision capabilities

---

## 💻 Hardware

MAX.AI is primarily being developed with **local inference** in mind.

The project is designed to experiment with running relatively capable language models on consumer hardware rather than relying entirely on expensive cloud APIs.

This makes MAX.AI suitable for experimentation with:

* Local AI
* LLM inference
* Quantized models
* GPU acceleration
* CPU inference
* AI optimization

---

## 🔐 Privacy

One of the long-term goals of MAX.AI is **local-first AI**.

Running the model locally means conversations can potentially remain on the user's own machine instead of being sent to a third-party AI provider.

> Privacy characteristics depend on the specific MAX.AI configuration and features being used.

---

## 🛠️ Technology

MAX.AI is an evolving project, so its technology stack may change.

Current / experimental technologies include:

| Technology         | Purpose                     |
| ------------------ | --------------------------- |
| 🤖 Qwen3.5 9B      | Language model              |
| 📦 Q4_K_M          | Model quantization          |
| 💻 Local inference | AI execution                |
| 🧠 LLM             | Conversational intelligence |

More technologies will be documented as the project stabilizes.

---

## 📸 Screenshots

> Screenshots will be added as the MAX.AI interface develops.

---

## 🧪 Experimental

MAX.AI is currently an **experimental project**.

Things may break.

Things may change.

Some features may disappear.

Some features may be completely rewritten.

That's part of the development process.

---

## 📌 Project Philosophy

MAX.AI is built around a simple idea:

> **Build it. Break it. Understand it. Improve it.**

Instead of treating AI as a black box, MAX.AI is also a learning project for exploring:

* Large Language Models
* AI inference
* Model quantization
* Prompt engineering
* AI application development
* Local AI infrastructure
* Performance optimization
* Human-AI interaction

---

## 🚀 Future Vision

The long-term vision for MAX.AI is to become more than a simple chatbot.

```text
                 ┌────────────────────┐
                 │      MAX.AI        │
                 │   Personal AI      │
                 └─────────┬──────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
       Chat             Memory            Tools
          │                │                │
       Vision            RAG             Coding
          │                │                │
       Voice          Knowledge         Automation
          │                │                │
          └────────────────┼────────────────┘
                           │
                    Local / Private
                         AI Core
```

The ultimate goal is to create a **personal AI ecosystem** that can understand context, remember information, use tools, interact with files, assist with development, and operate locally.

---

## ⚠️ Disclaimer

MAX.AI is currently under active development.

It may contain bugs, incomplete features, performance issues, and experimental code.

**Do not rely on MAX.AI for critical decisions or production workloads at this stage.**

---

## 👨‍💻 Developer

**Michael AP**

Building MAX.AI as an ongoing experiment in local AI, software engineering, and machine learning.

---

## ⭐ Support

If you find the project interesting, consider giving the repository a ⭐ on GitHub.

Every star helps motivate further development.

---

<p align="center">

**MAX.AI — Built to learn. Built to evolve.**

</p>
