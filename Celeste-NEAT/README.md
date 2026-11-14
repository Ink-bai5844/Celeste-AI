# Celeste NEAT

An attempt at making an AI learn to play the game [Celeste](https://store.steampowered.com/app/504230/Celeste/) using [Neuroevolution of Augmenting Topologies (NEAT)](https://en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies), specifically [neat-python](https://neat-python.readthedocs.io/en/latest/).
It uses data fetched directly from the game engine at runtime using [this instrumentation framework](https://github.com/hdrien0/Celeste-Instrumentation).

Thanks to @zteboui for the original idea and the help!

## Overview

Video demo :

[![video demo](https://img.youtube.com/vi/SePBjJ5Z0Lo/0.jpg)](https://www.youtube.com/watch?v=SePBjJ5Z0Lo)


The AI agents are neural networks. They are grouped in generations, and the better-performing agents of each generation are combined and mutated to form the next one. 
At every frame of the game, the agents decide what actions to take based on the following information :
- The distance to the nearest object in 8 directions
- The type of the nearest object for those 8 directions
- Their X and Y velocity
- Their ability to dash
- Their amount of stamina remaining
- The angle to their objective
- The distance to their objective  

The calculation of the agents' performance is based on their distance to their objective (usually the exit of the room) and how fast they reach it.

### Performance

> The NEAT algorithm often arrives at effective networks more quickly than other contemporary neuro-evolutionary techniques and reinforcement learning methods (Wikipedia)  

NEAT can evolve agents that clear simple rooms them in very few generations (~10 generations of 150 agents for the first room). They are even able to do it quite swiftly and to use some advanced movement techniques!  

However, the current simplicity of the agents' input data makes it unviable for levels requiring greater anticipation or complex interactions with the surroundings. With that said, there is still a lot of room for improvement, and other learning techniques such as using Inverse Reinforcement Learning on human gameplay could be well-suited to the task. 


## Installation

- Follow the steps to install the instrumentation part of the project [here](https://github.com/hdrien0/Celeste-Instrumentation#installation).
- Clone this repository and navigate to the repo folder

### Linux / MacOS
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Usage

After launching the instrumented game :
```bash
python celeste_neat.py
```
See `celest_neat_config` for the NEAT parameters.

The default training stack now uses an enhanced state vector plus route-plan hints, for 146 neural-network inputs:

* 32 raycast distances around Madeline
* 32 raycast object types
* a 7x7 local terrain grid around the player
* velocity, dash, ground, stamina, objective, progress, stuck, and speed features
* 13 route-plan features from the room planner, such as next waypoint delta, waypoint distance, route progress, confidence, and action hints

The game sends 137 floats each frame. The first 133 are local state features, and the final 4 are metadata used for scoring. The Python side appends 13 planner features before sending inputs to NEAT.

Useful training options:

```bash
python celeste_neat.py --no-visualize
python celeste_neat.py --curriculum --episodes-per-genome 2 --no-visualize
python celeste_neat.py --network recurrent --no-visualize
```

`--curriculum` rotates through several first-room waypoints so the population can learn short movement goals before the full exit objective. `--episodes-per-genome` evaluates each genome more than once and averages the score, which improves robustness at the cost of runtime.

### LLM route planner

When a room starts, the instrumented game sends a compact JSON map summary to Python. The summary includes room bounds, objective, a coarse whole-room tile grid, and semantic entity categories. Python converts that into route hints for the local NEAT controller.

Without an API key, the planner uses a deterministic fallback route. To call an OpenAI-compatible Chat Completions API, edit `llm_config.json`:

```json
{
  "enabled": true,
  "api_key": "sk-your-api-key-here",
  "model": "gpt-4o-mini",
  "base_url": "https://api.openai.com/v1",
  "failure_threshold": 5,
  "timeout": 120,
  "display_plans": true,
  "max_tokens": 4096,
  "retries": 1,
  "retry_delay": 2,
  "response_format": true
}
```

Then run:

```bash
python celeste_neat.py --llm-planner --curriculum --episodes-per-genome 2 --no-visualize
```

For compatible providers, change `base_url` and `model` in `llm_config.json`:

```json
{
  "enabled": true,
  "api_key": "your-provider-key",
  "model": "your-model-name",
  "base_url": "https://your-provider.example/v1",
  "failure_threshold": 5,
  "timeout": 120,
  "display_plans": true,
  "max_tokens": 4096,
  "retries": 1,
  "retry_delay": 2,
  "response_format": true
}
```

The default config path is `llm_config.json`. You can pass another file with `--llm-config path/to/config.json`. Environment variables like `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `CELESTE_LLM_MODEL` still work as temporary overrides.

When `display_plans` is `true`, every newly generated or revised route plan is printed in the training terminal, including source, map key, strategy, confidence, waypoints, hazards, and notes. Set it to `false` to silence plan output, or temporarily override it with `CELESTE_LLM_DISPLAY_PLANS=false`.

`timeout` is the per-request wait time in seconds. `retries` controls how many extra attempts are made after the first request fails, and `retry_delay` is the wait between attempts. `max_tokens` caps the route-plan response length.

`response_format` asks compatible providers to return a JSON object. If a provider returns empty content or does not support that option, set it to `false`; the planner will still extract a JSON object from plain text or a ```json code block.

After repeated failures on the same room, the planner includes recent failure positions, death/stuck state, score, and the previous waypoints in the next LLM request so the route can be revised.

### Parallel evaluation

You can evaluate genomes in parallel by running several instrumented Celeste instances and giving each instance a different socket port.

For example, to use 4 workers, start Python first:

```bash
python celeste_neat.py --workers 4 --base-port 25001 --no-visualize
```

Then run 4 patched Celeste copies. Each copy must have its own `InstrumentationParameters.xml` next to `Celeste.exe`, with a different `ConnectionPort`:

```xml
<ConnectionPort>25001</ConnectionPort>
<ConnectionPort>25002</ConnectionPort>
<ConnectionPort>25003</ConnectionPort>
<ConnectionPort>25004</ConnectionPort>
```

The ports can also be specified explicitly:

```bash
python celeste_neat.py --ports 25001,25011,25021 --no-visualize
```
