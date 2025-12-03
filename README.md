# create_rl
Generative Design using Reinforcement Learning

## Installation/Running with Docker
Ensure you have docker, docker compose and docker-nivida-toolkit installed.

```
docker compose up dev -d
docker exec -it gen_rl-dev-1 bash
```

You can also set up VSCode to now work within this container.

## Running with uv
Not tested as much, may lack some system dependecies that come with Docker base image.

```
uv venv --python 3.12 --seed
uv pip install -e gen_rl
```

## FAQ
### Why are there X11 forwarding in the `docker-compose`?
We need those to see the visualization inside docker. This is not needed for the web-based GUI.