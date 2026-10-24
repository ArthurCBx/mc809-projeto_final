import grpc
import argparse
import threading
import proto_config.message_pb2 as pb2
import proto_config.message_pb2_grpc as pb2_grpc
from concurrent import futures
import logging
import numpy as np


class ServerConsensus(pb2_grpc.FedMDLearningServicer):
    """Servidor de consenso do FedMD.

    Cada chamada de SendClientLogits fica bloqueada até todos os clientes esperados enviarem
    seus logits da rodada. A última chamada calcula a média e libera todas as outras, que
    respondem com o mesmo consenso.
    """

    def __init__(self, expected_clients: int=3, round_timeout: float=600):
        self.expected_clients = expected_clients
        self.round_timeout    = round_timeout

        self.condition           = threading.Condition()  # o gRPC atende cada cliente em uma thread
        self.pending_predictions = {}                     # client_id -> logits achatados
        self.round               = 0
        self.consensus           = None

    def SendClientLogits(self, request, context):
        with self.condition:
            if request.client_id in self.pending_predictions:
                context.abort(grpc.StatusCode.ALREADY_EXISTS,
                              f'Cliente {request.client_id} já enviou logits na rodada {self.round}.')

            my_round = self.round
            self.pending_predictions[request.client_id] = np.array(request.logits, dtype=np.float32)
            logging.info(f'Rodada {my_round}: logits do cliente {request.client_id} '
                         f'({len(self.pending_predictions)}/{self.expected_clients}).')

            if len(self.pending_predictions) == self.expected_clients:
                self._close_round()
            elif not self.condition.wait_for(lambda: self.round != my_round, timeout=self.round_timeout):
                del self.pending_predictions[request.client_id]
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED,
                              f'Rodada {my_round} não recebeu os logits de todos os clientes a tempo.')

            if self.error:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, self.error)
            return pb2.ServerConsensus(consensus_logits=self.consensus.tolist())

    def _close_round(self):
        """Calcula o consenso da rodada e acorda os clientes que estão esperando. Chamar com o lock."""
        # Média posição a posição: todos os clientes precisam ter usado o mesmo public_idx, na mesma ordem.
        self.consensus = np.mean(np.stack(list(self.pending_predictions.values())), axis=0)
        logging.info(f'Rodada {self.round} encerrada.' + (f' Erro: {self.error}' if self.error else ''))

        self.pending_predictions = {}  # Reset for next round
        self.round += 1
        self.condition.notify_all()


def serve(expected_clients: int):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    logging.info("Starting server...")

    # Cada cliente ocupa uma thread enquanto espera o consenso; com menos threads que clientes, trava.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=expected_clients + 2))
    pb2_grpc.add_FedMDLearningServicer_to_server(ServerConsensus(expected_clients), server)
    server.add_insecure_port(f'[::]:50051')
    server.start()
    logging.info(f"Server started on port 50051, waiting for {expected_clients} clients per round.")
    server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clients', type=int, default=3, help='Número de clientes por rodada.')
    args = parser.parse_args()
    serve(args.clients)