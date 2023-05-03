import copy
import math
import warnings
from typing import List, Dict, Union, Callable, Tuple, Iterable

import numpy as np
import torch
from crp.attribution import CondAttribution
from crp.concepts import Concept
from crp.concepts import ChannelConcept
from crp.helper import load_maximization, load_statistics
from crp.hooks import FeatVisHook
from crp.image import vis_img_heatmap
from crp.maximization import Maximization as Maximization
from crp.statistics import Statistics as Statistics
from crp.visualization import FeatureVisualization
from tqdm import tqdm
from zennit.composites import NameMapComposite
from zennit.core import Composite

class CondAttributionLocalization(CondAttribution):

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.take_prediction = 0
    def relevance_init(self, prediction, target_list, init_rel):

        if target_list:
            r = torch.zeros_like(prediction).to(self.device)
            for i, target in enumerate(target_list):
                if prediction[i].shape[0] == 0:
                    print("no predicted boxes")
                else:
                    k = min(self.take_prediction + 1, prediction[i].shape[0])
                    best_bb_id = torch.topk(prediction[i], k, dim=0).indices[k - 1, target].item()
                    if self.take_prediction != 0:
                        print("taking prediction num. ", k - 1, " (wanted ", self.take_prediction, ")")
                    r[i, best_bb_id, target] = torch.ones_like(r[i, best_bb_id, target]) * (
                            prediction[i, best_bb_id, target] > 0.25)
            init_rel = r / (r.sum() + 1e-12)
        else:
            prediction = prediction.clamp(min=0)

        return CondAttribution.relevance_init(self, prediction, None, init_rel)


class CondAttributionSegmentation(CondAttribution):

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.mask = 1
        self.rel_init = "zplus"

    def relevance_init(self, prediction, target_list, init_rel):
        # print("initil")
        if target_list:
            r = torch.zeros_like(prediction).to(self.device)
            pred = torch.nn.functional.softmax(prediction, dim=1)
            for i, target in enumerate(target_list):
                if isinstance(target, List):
                    assert len(target) == 1
                    target = target[0]
                argmax = (torch.argmax(pred, dim=1) == target)[i]
                expl = prediction
                # print(target_list, self.rel_init)
                if "zplus" in self.rel_init:
                    expl = expl.clamp(min=0)
                if "ones" in self.rel_init:
                    expl = torch.ones_like(expl)
                elif "prob" in self.rel_init:
                    expl = pred
                if "grad" in self.rel_init:
                    expl = expl / (prediction + 1e-10)
                r[i, target, :, :] = expl[i, target, :, :] * argmax
                r = r * self.mask
            init_rel = r / (r.sum() + 1e-12)
        else:
            prediction = prediction.clamp(min=0)

        return CondAttribution.relevance_init(self, prediction, None, init_rel)


class FeatureVisualizationMultiTarget(FeatureVisualization):

    def __init__(self, attribution: CondAttribution, dataset, layer_map: Dict[str, Concept],
                 preprocess_fn: Callable = None, max_target="sum", abs_norm=True, path="FeatureVisualization",
                 device=None):

        super().__init__(attribution, dataset, layer_map, preprocess_fn, max_target, abs_norm, path, device)

        self.RelMax = Maximization("relevance", "sum", abs_norm, path)
        self.ActMax = Maximization("activation", max_target, abs_norm, path)
        self.RelStats = Statistics("relevance", "sum", abs_norm, path)
        self.ActStats = Statistics("activation", max_target, abs_norm, path)

    def multitarget_to_single(self, multi_target):
        multi_target = np.array(multi_target).astype(int)
        single_targets = np.argwhere(multi_target == 1).reshape(-1)
        return single_targets

    def get_max_reference(
            self, concept_ids: Union[int, list], layer_name: str, mode="relevance", r_range: Tuple[int, int] = (0, 8),
            composite: Composite = None,
            rf=False, plot_fn=vis_img_heatmap, batch_size=32) -> Dict:
        """
        Retreive reference samples for a list of concepts in a layer. Relevance and Activation Maximization
        are availble if FeatureVisualization was computed for the mode. In addition, conditional heatmaps can be computed on reference samples.
        If the crp.concept class (supplied to the FeatureVisualization layer_map) implements masking for a single neuron in the 'mask_rf' method,
        the reference samples and heatmaps can be cropped using the receptive field of the most relevant or active neuron.

        Parameters:
        ----------
        concept_ids: int or list
        layer_name: str
        mode: "relevance" or "activation"
            Relevance or Activation Maximization
        r_range: Tuple(int, int)
            Range of N-top reference samples. For example, (3, 7) corresponds to the Top-3 to -6 samples.
            Argument must be a closed set i.e. second element of tuple > first element.
        composite: zennit.composites or None
            If set, compute conditional heatmaps on reference samples. `composite` is used for the CondAttribution object.
        rf: boolean
            If True, compute the CRP heatmap for the most relevant/most activating neuron only to restrict the conditonal heatmap
            on the receptive field.
        plot_fn: callable function with signature (samples: torch.Tensor, heatmaps: torch.Tensor, rf: boolean) or None
            Draws reference images. The function receives as input the samples used for computing heatmaps before preprocessing
            with self.preprocess_data and the final heatmaps after computation. In addition, the boolean flag 'rf' is passed to it.
            The return value of the function should correspond to the Cache supplied to the FeatureVisualization object (if available).
            If None, the raw tensors are returned.
        batch_size: int
            If heatmap is True, describes maximal batch size of samples to compute for conditional heatmaps.

        Returns:
        -------
        ref_c: dictionary.
            Key values correspond to channel index and values are reference samples. The values depend on the implementation of
            the 'plot_fn'.
        """

        ref_c = {}
        if not isinstance(concept_ids, Iterable):
            concept_ids = [concept_ids]

        if mode == "relevance":
            d_c_sorted, rel_c_sorted, rf_c_sorted = load_maximization(self.RelMax.PATH, layer_name)
        elif mode == "activation":
            d_c_sorted, rel_c_sorted, rf_c_sorted = load_maximization(self.ActMax.PATH, layer_name)
        else:
            raise ValueError("`mode` must be `relevance` or `activation`")

        if rf and not composite:
            warnings.warn(
                "The receptive field is only computed, if you fill the 'composite' argument with a zennit Composite.")

        for c_id in concept_ids:

            d_indices = d_c_sorted[r_range[0]:r_range[1], c_id]
            r_values = rel_c_sorted[r_range[0]:r_range[1], c_id]
            n_indices = rf_c_sorted[r_range[0]:r_range[1], c_id]

            if mode == "relevance":
                data_batch, targets_multi = self.get_data_concurrently(d_indices, preprocessing=True)
                data_batch_unprocessed, _ = self.get_data_concurrently(d_indices, preprocessing=False)
                targets_single = []
                for i_t, target in enumerate(targets_multi):
                    single_targets = self.multitarget_to_single(target)
                    for st in single_targets:
                        targets_single.append(st)

                targets = np.zeros(r_range[1] - r_range[0]).astype(int)
                for t in np.arange(self.dataset.class_names.__len__()):
                    try:
                        target_stats = load_statistics(self.RelStats.PATH, layer_name, t)
                        td_indices = target_stats[0][:, c_id]
                        tr_values = target_stats[1][:, c_id]
                        cond = [True if (x in td_indices) and (tr_values[list(td_indices).index(x)] == r) else False
                                for x, r in
                                zip(d_indices, r_values)]
                        targets[cond] = int(t)
                    except FileNotFoundError:
                        continue

                heatmaps = self._attribution_on_reference(data_batch, c_id, layer_name, composite, rf, n_indices,
                                                          batch_size, targets)

                if callable(plot_fn):
                    ref_c[c_id] = plot_fn(data_batch_unprocessed.detach(), heatmaps.detach(), rf)
                else:
                    ref_c[c_id] = data_batch_unprocessed.detach().cpu(), heatmaps.detach().cpu()

            else:
                ref_c[c_id] = self._load_ref_and_attribution(d_indices, c_id, n_indices, layer_name, composite, rf,
                                                             plot_fn, batch_size)

        return ref_c

    def run_distributed(self, composite: Composite, data_start, data_end, batch_size=16, checkpoint=500,
                        on_device=None):
        """
        max batch_size = max(multi_targets) * data_batch
        data_end: exclusively counted
        """

        self.saved_checkpoints = {"r_max": [], "a_max": [], "r_stats": [], "a_stats": []}
        last_checkpoint = 0

        n_samples = data_end - data_start
        samples = np.arange(start=data_start, stop=data_end)

        if n_samples > batch_size:
            batches = math.ceil(n_samples / batch_size)
        else:
            batches = 1
            batch_size = n_samples

        # feature visualization is performed inside forward and backward hook of layers
        name_map, dict_inputs = [], {}
        for l_name, concept in self.layer_map.items():
            hook = FeatVisHook(self, concept, l_name, dict_inputs, on_device)
            name_map.append(([l_name], hook))
        fv_composite = NameMapComposite(name_map)

        if composite:
            composite.register(self.attribution.model)
        fv_composite.register(self.attribution.model)

        pbar = tqdm(total=batches, dynamic_ncols=True)

        for b in range(batches):

            pbar.update(1)

            samples_batch = samples[b * batch_size: (b + 1) * batch_size]
            data_batch, targets_samples = self.get_data_concurrently(samples_batch, preprocessing=True)

            targets_samples = np.array(targets_samples)  # numpy operation needed

            # convert multi target to single target if user defined the method
            data_broadcast, targets, sample_indices = [], [], []
            try:
                for i_t, target in enumerate(targets_samples):
                    single_targets = self.multitarget_to_single(target)
                    for st in single_targets:
                        targets.append(st)
                        data_broadcast.append(data_batch[i_t])
                        sample_indices.append(samples_batch[i_t])
                if len(data_broadcast) == 0:
                    continue
                # TODO: test stack
                data_broadcast = torch.stack(data_broadcast, dim=0)
                sample_indices = np.array(sample_indices)
                targets = np.array(targets)

            except NotImplementedError:
                data_broadcast, targets, sample_indices = data_batch, targets_samples, samples_batch

            conditions = [{self.attribution.MODEL_OUTPUT_NAME: [t]} for t in targets]
            # dict_inputs is linked to FeatHooks
            if n_samples > batch_size:
                batches_ = math.ceil(len(conditions) / batch_size)
            else:
                batches_ = 1

            for b_ in range(batches_):
                data_broadcast_ = data_broadcast[b_ * batch_size: (b_ + 1) * batch_size]
                # print(len(conditions), len(data_broadcast_))
                conditions_ = conditions[b_ * batch_size: (b_ + 1) * batch_size]
                # dict_inputs is linked to FeatHooks
                dict_inputs["sample_indices"] = sample_indices[b_ * batch_size: (b_ + 1) * batch_size]
                dict_inputs["targets"] = targets[b_ * batch_size: (b_ + 1) * batch_size]

                # composites are already registered before
                self.attribution(data_broadcast_, conditions_, None, exclude_parallel=False)

            if b % checkpoint == checkpoint - 1:
                self._save_results((last_checkpoint, sample_indices[-1] + 1))
                last_checkpoint = sample_indices[-1] + 1

        if len(sample_indices):
            self._save_results((last_checkpoint, sample_indices[-1] + 1))

        if composite:
            composite.remove()
        fv_composite.remove()

        pbar.close()

        return self.saved_checkpoints

    def _attribution_on_reference(self, data, concept_id: int, layer_name: str, composite, rf=False,
                                  neuron_ids: list = [], batch_size=32, targets=None):

        n_samples = len(data)
        if n_samples > batch_size:
            batches = math.ceil(n_samples / batch_size)
        else:
            batches = 1
            batch_size = n_samples

        if rf and (len(neuron_ids) != n_samples):
            raise ValueError("length of 'neuron_ids' must be equal to the length of 'data'")

        heatmaps = []
        for b in range(batches):
            data_batch = data[b * batch_size: (b + 1) * batch_size].detach().requires_grad_()

            if targets is None:
                start_layer = layer_name
                exlude_parallel = False
            else:
                targets_batch = targets[b * batch_size: (b + 1) * batch_size]
                start_layer = None
                exlude_parallel = True

            if rf:
                neuron_ids_ = neuron_ids[b * batch_size: (b + 1) * batch_size]
                if targets is None:
                    conditions = [{layer_name: {concept_id: n_index}} for n_index in neuron_ids_]
                else:
                    conditions = [{layer_name: {concept_id: n_index}, "y": t} for n_index, t in
                                  zip(neuron_ids_, targets_batch)]
                attr = self.attribution(data_batch, conditions, composite, mask_map=ChannelConcept.mask_rf,
                                        start_layer=start_layer, on_device=self.device,
                                        exclude_parallel=exlude_parallel)
            else:
                if targets is None:
                    conditions = [{layer_name: [concept_id]}]
                else:
                    conditions = [{layer_name: [concept_id], "y": t} for t in targets_batch]
                # initialize relevance with activation before non-linearity (could be changed in a future release)
                attr = self.attribution(data_batch, conditions, composite, start_layer=start_layer,
                                        on_device=self.device, exclude_parallel=exlude_parallel)

            heatmaps.append(attr.heatmap)

        return torch.cat(heatmaps, dim=0)


class FeatureVisualizationLocalization(FeatureVisualizationMultiTarget):
    def get_data_sample(self, index, preprocessing=True) -> Tuple[torch.tensor, List[int]]:
        """
        returns a data sample from dataset at index.

        Parameter:
            index: integer
            preprocessing: boolean.
                If True, return the sample after preprocessing. If False, return the sample for plotting.
        """

        data, target = self.dataset[index]

        if not preprocessing:
            data = self.dataset.reverse_normalization(data)

        target = target[..., 1].long()
        targets = np.unique(target)
        if len(targets) == 0:
            print(index)
        targets = np.random.permutation(targets)
        targets = [1 if (i in targets.astype(int)) else 0 for i in range(len(self.dataset.class_names))]

        return data.unsqueeze(0).to(self.device).requires_grad_(), targets


class FeatureVisualizationSegmentation(FeatureVisualizationMultiTarget):
    def get_data_sample(self, index, preprocessing=True) -> Tuple[torch.tensor, List[int]]:
        """
        returns a data sample from dataset at index.

        Parameter:
            index: integer
            preprocessing: boolean.
                If True, return the sample after preprocessing. If False, return the sample for plotting.
        """

        data, target = self.dataset[index]

        if not preprocessing:
            data = self.dataset.reverse_normalization(data)

        targets = torch.unique(target)
        targets = torch.Tensor([t for t in targets if t != 255]).int()  # no background class
        targets = targets[torch.randperm(len(targets))]

        # print(data.shape)
        targets = [1 if (i in list(targets.numpy())) else 0 for i in range(len(self.dataset.class_names))]
        targets = np.array(targets).flatten().astype(int)
        return data.unsqueeze(0).to(self.device).requires_grad_(), targets
