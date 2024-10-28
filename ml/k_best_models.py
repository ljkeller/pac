import heapq
import logging

logger = logging.getLogger(__name__)


class KBestModels:
    """
    Heap to store the best models by accuracy.

    Especially useful for a k-fold cross validation
    """

    def __init__(self, max_size):
        self.max_size = max_size
        self.heap = []

    def push(self, model, acc):
        """Push a model and its accuracy onto the heap"""

        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, (acc, model))
        elif acc > self.heap[0][0]:
            heapq.heappushpop(self.heap, (acc, model))
            logger.debug(
                f"Found better model with accuracy: {acc*100:.2f}%, popping lowest acc model."
            )

    def get_best_acc_model_pair(self):
        """Get the best (accuracy, model) pair from the heap

        Returns:
            Tuple[None, None], or Tuple[float, torch.nn.Module]: The best accuracy and model pair
        """
        return heapq.nlargest(1, self.heap)[0] if self.heap else (None, None)

    def __iter__(self):
        """Iterate over the heap, yielding the (accuracy, model) pairs in descending order of accuracy"""

        for pair in heapq.nlargest(len(self.heap), self.heap):
            yield pair
