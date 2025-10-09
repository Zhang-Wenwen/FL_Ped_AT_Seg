from nvflare.app_common.aggregators import InTimeAccumulateWeightedAggregator
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.dxo import DXO, DataKind
from loguru import logger


class CustomizedAggregator(InTimeAccumulateWeightedAggregator):
    def __init__(self, exclude_vars,aggregation_weights, expected_data_kind, weigh_by_local_iter: bool = True):
        super().__init__(exclude_vars,aggregation_weights, expected_data_kind, weigh_by_local_iter)

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Called when workflow determines to generate shareable to send back to contributors

        Args:
            fl_ctx (FLContext): context provided by workflow

        Returns:
            Shareable: the weighted mean of accepted shareables from contributors
        """

        self.log_debug(fl_ctx, "Start aggregation")
        result_dxo_dict = dict()

        for key in self.expected_data_kind.keys():
            aggregated_dxo = self.dxo_aggregators[key].aggregate(fl_ctx)
            if key == self._single_dxo_key:  # return single DXO with aggregation results
                return aggregated_dxo.to_shareable()
            self.log_info(fl_ctx, f"Aggregated contributions matching key '{key}'.")
            result_dxo_dict.update({key: aggregated_dxo})
        # return collection of DXOs with aggregation results
        collection_dxo = DXO(data_kind=DataKind.COLLECTION, data=result_dxo_dict)
        return collection_dxo.to_shareable()
