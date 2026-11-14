import argparse
import copy
import os
import pickle
import queue
import sys
import threading

import neat

LOCAL_INTERFACE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Celeste-Instrumentation", "CelestePythonInterface"))
if LOCAL_INTERFACE_PATH not in sys.path:
    sys.path.insert(0, LOCAL_INTERFACE_PATH)

from visualize import *
from CelestePythonInterface import SocketInterface, SessionParameters, SocketServer
from llm_planner import CelesteRoutePlanner
from utils import inputs_preprocessor, score_function
from FileReporter import *


class CelesteNeat:

    def __init__(
        self,
        session_parameters,
        neat_config_file,
        output_folder,
        restore_checkpoint=False,
        restore_file=None,
        host="127.0.0.1",
        ports=None,
        generations=500,
        visualize_results=True,
        network_type="feedforward",
        episodes_per_genome=1,
        curriculum=None,
        route_planner=None,
    ):
        self.client_sockets = []
        self.restore_checkpoint = restore_checkpoint
        self.restore_file = restore_file
        self.session_parameters = session_parameters
        self.output_folder = output_folder
        self.host = host
        self.ports = ports or [25001]
        self.generations = generations
        self.visualize_results = visualize_results
        self.network_type = network_type
        self.episodes_per_genome = episodes_per_genome
        self.curriculum = curriculum or [session_parameters]
        self.route_planner = route_planner or CelesteRoutePlanner.from_env()
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            neat_config_file,
        )

    def await_game_connections(self):
        servers = []
        for port in self.ports:
            servers.append((port, SocketServer(self.host, port)))

        print(f"Waiting for {len(servers)} Celeste instance(s)...")
        client_sockets = []
        for worker_index, (port, socket_server) in enumerate(servers):
            print(f"Worker {worker_index}: waiting on {self.host}:{port}")
            client_socket, addr = socket_server.await_connection()
            print(f"Worker {worker_index}: connected from {addr}")
            client_sockets.append(client_socket)

        self.client_sockets = client_sockets

    def create_network(self, genome, config):
        if self.network_type == "recurrent":
            return neat.nn.RecurrentNetwork.create(genome, config)
        return neat.nn.FeedForwardNetwork.create(genome, config)

    def session_for_episode(self, genome_id, episode_index):
        curriculum_index = (genome_id + episode_index) % len(self.curriculum)
        return self.curriculum[curriculum_index]

    def eval_genome(self, genome_id, genome, config, client_socket):
        episode_scores = []
        input_count = config.genome_config.num_inputs
        for episode_index in range(self.episodes_per_genome):
            net = self.create_network(genome, config)
            session_parameters = self.session_for_episode(genome_id, episode_index)
            inter = SocketInterface(
                client_socket,
                net.activate,
                session_parameters,
                lambda state, room_plan=None, input_count=input_count: inputs_preprocessor(
                    state,
                    input_count,
                    room_plan,
                    self.route_planner,
                ),
                route_planner=self.route_planner,
            )
            final_state = inter.run()
            score = score_function(final_state, session_parameters)
            self.route_planner.record_result(inter.active_plan, final_state, score)
            episode_scores.append(score)
        return sum(episode_scores) / len(episode_scores)

    def eval_genomes(self, genomes, config):
        if len(self.client_sockets) == 1:
            for genome_id, genome in genomes:
                genome.fitness = self.eval_genome(genome_id, genome, config, self.client_sockets[0])
            return

        work_queue = queue.Queue()
        error_queue = queue.Queue()
        stop_event = threading.Event()

        for genome_id, genome in genomes:
            work_queue.put((genome_id, genome))

        def worker(worker_index, client_socket):
            while not stop_event.is_set():
                try:
                    genome_id, genome = work_queue.get_nowait()
                except queue.Empty:
                    return

                try:
                    genome.fitness = self.eval_genome(genome_id, genome, config, client_socket)
                except Exception as exc:
                    genome.fitness = float("-inf")
                    stop_event.set()
                    error_queue.put((worker_index, genome_id, exc))
                finally:
                    work_queue.task_done()

        threads = []
        for worker_index, client_socket in enumerate(self.client_sockets):
            thread = threading.Thread(
                target=worker,
                args=(worker_index, client_socket),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if not error_queue.empty():
            worker_index, genome_id, exc = error_queue.get()
            raise RuntimeError(
                f"Worker {worker_index} failed while evaluating genome {genome_id}. "
                "Check that every Celeste instance is still running and connected."
            ) from exc

    def run(self):

        # Create the population, which is the top-level object for a NEAT run.
        if not self.restore_checkpoint:
            p = neat.Population(self.neat_config)
        else:
            p = neat.Checkpointer.restore_checkpoint(self.restore_file)

        # p.add_reporter(FileReporter(self.output_folder+'/', True))
        # see the FileReporter class in FileReporter.py to see what it does
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.StdOutReporter(True))
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        p.add_reporter(neat.Checkpointer(1, 10000, self.output_folder + "/neat-checkpoint-"))

        self.await_game_connections()
        winner = p.run(self.eval_genomes, self.generations)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))
        with open(os.path.join(self.output_folder, "winner.pkl"), "wb") as winner_file:
            pickle.dump(winner, winner_file)

        if self.visualize_results:
            node_names = {-1: 'D_E', -2: 'D_NE', -3: 'D_N', -4: 'D_NW', -5: 'D_W', -6: 'D_SW', -7: 'D_S', -8: 'D_SE', -9: 'T_E', -10: 'T_NE', -11: 'T_N', -12: 'T_NW', -13: 'T_W', -14: 'T_SW', -15: 'T_S', -16: 'T_SE', -17: 'X_VELOCITY', -18: 'Y_VELOCITY', -19: 'CAN_DASH', -20: 'ON_GROUND', -21: 'STAMINA', -22: 'ANGLE_TO_OBJ', -23: 'DIST_TO_OBJ', 0: 'RIGHT', 1: 'LEFT', 2: 'UP', 3: 'DOWN', 4: 'JUMP', 5: 'DASH', 6: 'GRAB'}
            draw_net(self.neat_config, winner, True, node_names=node_names)
            draw_net(self.neat_config, winner, True, node_names=node_names, prune_unused=True)
            plot_stats(stats, ylog=False, view=True)
            plot_species(stats, view=True)


def parse_ports(args):
    if args.ports:
        ports = [int(port.strip()) for port in args.ports.split(",") if port.strip()]
        if not ports:
            raise ValueError("--ports must contain at least one port")
        return ports

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    return [args.base_port + offset for offset in range(args.workers)]


def clone_session_parameters(session_parameters, **overrides):
    cloned = copy.deepcopy(session_parameters)
    for key, value in overrides.items():
        setattr(cloned, key, value)
    return cloned


def build_curriculum(base_session_parameters, enabled=False, timeout_seconds=None):
    if not enabled:
        return [base_session_parameters]

    # Conservative first-room curriculum: same room, several nearby waypoints.
    # It teaches progress and jumping before asking the agent to solve the full exit path.
    if timeout_seconds is None:
        timeouts = [3, 4, 5]
    else:
        timeouts = [timeout_seconds, timeout_seconds, timeout_seconds]
    return [
        clone_session_parameters(base_session_parameters, ObjectiveXCoordinate=180, ObjectiveYCoordinate=-24, TimeoutSeconds=timeouts[0]),
        clone_session_parameters(base_session_parameters, ObjectiveXCoordinate=230, ObjectiveYCoordinate=-24, TimeoutSeconds=timeouts[1]),
        clone_session_parameters(base_session_parameters, ObjectiveXCoordinate=282, ObjectiveYCoordinate=-24, TimeoutSeconds=timeouts[2]),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Celeste agents with NEAT.")
    parser.add_argument("-j", "--workers", type=int, default=1, help="number of Celeste instances to evaluate genomes in parallel")
    parser.add_argument("--host", default="127.0.0.1", help="host/interface used by the Python socket servers")
    parser.add_argument("--base-port", type=int, default=25001, help="first port to listen on when --ports is not provided")
    parser.add_argument("--ports", help="comma-separated explicit ports, for example: 25001,25002,25003")
    parser.add_argument("--generations", type=int, default=500, help="maximum number of generations to run")
    parser.add_argument("--output-folder", default="run-0", help="folder for NEAT checkpoints")
    parser.add_argument("--config", default="celeste_neat_config", help="NEAT config file")
    parser.add_argument("--restore-checkpoint", help="path to a neat-checkpoint-* file to resume from")
    parser.add_argument("--no-visualize", action="store_true", help="skip graph/stats windows after training")
    parser.add_argument("--network", choices=["feedforward", "recurrent"], default="feedforward", help="network implementation used to activate genomes")
    parser.add_argument("--episodes-per-genome", type=int, default=1, help="number of sessions used to score each genome")
    parser.add_argument("--timeout-seconds", type=float, default=None, help="maximum seconds per game session")
    parser.add_argument("--curriculum", action="store_true", help="rotate through several first-room waypoints while training")
    parser.add_argument("--continue-on-level-exit", action="store_true", help="keep the same session running after a room transition instead of ending and restarting the room")
    parser.add_argument("--llm-planner", action="store_true", help="enable OpenAI-compatible LLM route planning")
    parser.add_argument("--llm-config", default="llm_config.json", help="JSON file with LLM planner settings")
    parser.add_argument("--llm-model", help="model name for the OpenAI-compatible route planner")
    parser.add_argument("--llm-base-url", help="OpenAI-compatible API base URL, default: https://api.openai.com/v1")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    ports = parse_ports(args)

    session_parameters = SessionParameters()
    session_parameters.Level = "1"
    session_parameters.AreaKey = 1
    session_parameters.AreaMode = 0
    session_parameters.TimeoutSeconds = args.timeout_seconds if args.timeout_seconds is not None else 5
    session_parameters.ObjectiveXCoordinate = 282
    session_parameters.ObjectiveYCoordinate = -24
    if args.continue_on_level_exit:
        session_parameters.EndOnLevelExit = "false"
    curriculum = build_curriculum(session_parameters, args.curriculum, args.timeout_seconds)
    route_planner = CelesteRoutePlanner.from_config_file(args.llm_config)
    if args.llm_planner:
        route_planner.enabled = True
    if args.llm_model:
        route_planner.model = args.llm_model
    if args.llm_base_url:
        route_planner.base_url = args.llm_base_url.rstrip("/")

    celeste_neat = CelesteNeat(
        session_parameters,
        args.config,
        args.output_folder,
        restore_checkpoint=args.restore_checkpoint is not None,
        restore_file=args.restore_checkpoint,
        host=args.host,
        ports=ports,
        generations=args.generations,
        visualize_results=not args.no_visualize,
        network_type=args.network,
        episodes_per_genome=args.episodes_per_genome,
        curriculum=curriculum,
        route_planner=route_planner,
    )
    celeste_neat.run()
