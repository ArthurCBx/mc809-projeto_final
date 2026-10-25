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

    Com weighting='accuracy', a média é ponderada pela acurácia de cada cliente acima do acaso no
    subconjunto público, max(acc - 0.5, 0): um cliente que colapsou para uma classe só fica com
    peso ~0 e não contamina o consenso. Com weighting='uniform', é a média simples do FedMD.
    """

    def __init__(self, expected_clients: int=3, round_timeout: float=600, weighting: str='accuracy'):
        self.expected_clients = expected_clients
        self.round_timeout    = round_timeout
        self.weighting        = weighting

        self.condition           = threading.Condition()  # o gRPC atende cada cliente em uma thread
        self.pending_predictions = {}                     # client_id -> logits achatados
        self.pending_accuracies  = {}                     # client_id -> acurácia no subconjunto público
        self.round               = 0
        self.consensus           = None

    def SendClientLogits(self, request, context):
        with self.condition:
            if request.client_id in self.pending_predictions:
                context.abort(grpc.StatusCode.ALREADY_EXISTS,
                              f'Cliente {request.client_id} já enviou logits na rodada {self.round}.')

            my_round = self.round
            self.pending_predictions[request.client_id] = np.array(request.logits, dtype=np.float32)
            self.pending_accuracies[request.client_id]  = request.public_accuracy
            logging.info(f'Rodada {my_round}: logits do cliente {request.client_id} '
                         f'({len(self.pending_predictions)}/{self.expected_clients}).')

            if len(self.pending_predictions) == self.expected_clients:
                self._close_round()
            elif not self.condition.wait_for(lambda: self.round != my_round, timeout=self.round_timeout):
                del self.pending_predictions[request.client_id]
                del self.pending_accuracies[request.client_id]
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED,
                              f'Rodada {my_round} não recebeu os logits de todos os clientes a tempo.')

            return pb2.ServerConsensus(consensus_logits=self.consensus.tolist())

    def _close_round(self):
        """Calcula o consenso da rodada e acorda os clientes que estão esperando. Chamar com o lock."""
        client_ids = sorted(self.pending_predictions)
        if self.weighting == 'accuracy':
            weights = np.array([max(self.pending_accuracies[cid] - 0.5, 0.0) for cid in client_ids])
        else:
            weights = np.ones(len(client_ids))
        if weights.sum() == 0:
            weights = np.ones(len(client_ids))  # todos no acaso: nenhum é melhor que os outros
        weights = weights / weights.sum()

        # Média posição a posição: todos os clientes precisam ter usado o mesmo public_idx, na mesma ordem.
        logits = np.stack([self.pending_predictions[cid] for cid in client_ids])
        self.consensus = np.tensordot(weights, logits, axes=1)
        logging.info(f'Rodada {self.round} encerrada. Pesos: ' + ', '.join(
            f'cliente {cid} {w:.2f} (acc {self.pending_accuracies[cid]:.3f})' for cid, w in zip(client_ids, weights)))

        self.pending_predictions = {}  # Reset for next round
        self.pending_accuracies  = {}
        self.round += 1
        self.condition.notify_all()


def serve(expected_clients: int, weighting: str):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    logging.info("Starting server...")

    # Cada cliente ocupa uma thread enquanto espera o consenso; com menos threads que clientes, trava.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=expected_clients + 2))
    pb2_grpc.add_FedMDLearningServicer_to_server(ServerConsensus(expected_clients, weighting=weighting), server)
    server.add_insecure_port(f'[::]:50051')
    server.start()
    logging.info(f"Server started on port 50051, waiting for {expected_clients} clients per round, consensus weighting: {weighting}.")
    server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clients', type=int, default=3, help='Número de clientes por rodada.')
    parser.add_argument('--weighting', choices=['accuracy', 'uniform'], default='accuracy',
                        help="Pondera o consenso pela acurácia de cada cliente no público ('accuracy') ou usa a média simples ('uniform').")
    args = parser.parse_args()
    serve(args.clients, args.weighting)