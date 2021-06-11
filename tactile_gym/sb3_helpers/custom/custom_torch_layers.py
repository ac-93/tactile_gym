from typing import Dict, List, Tuple, Type, Union
import gym
import torch as th
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN, create_mlp


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for Dict observation spaces.
    Builds a feature extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param mlp_extractor_net_arch: Architecture for mlp encoding of state features before concatentation to cnn output
    :param mlp_activation_fn: Activation Func for MLP encoding layers
    :param cnn_output_dim: Number of features to output from each CNN submodule(s)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        mlp_extractor_net_arch: Union[int, List[int]] = None,  # List[Union[int, Dict[str, List[int]]]],
        mlp_activation_fn: Type[nn.Module] = nn.Tanh,
        cnn_output_dim: int = 64,
    ):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        cnn_extractors = {}
        flatten_extractors = {}

        cnn_concat_size = 0
        flatten_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                cnn_extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim)
                cnn_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                flatten_extractors[key] = nn.Flatten()
                flatten_concat_size += get_flattened_obs_dim(subspace)

        total_concat_size = cnn_concat_size + flatten_concat_size

        # default mlp arch to empty list if not specified
        if mlp_extractor_net_arch is None:
            mlp_extractor_net_arch = []

        for layer in mlp_extractor_net_arch:
            assert isinstance(layer, int), "Error: the mlp_extractor_net_arch can only include ints"

        # once vector obs is flattened can pass it through mlp
        if (mlp_extractor_net_arch != []) and (flatten_concat_size > 0):
            mlp_extractor = create_mlp(
                flatten_concat_size, mlp_extractor_net_arch[-1], mlp_extractor_net_arch[:-1], mlp_activation_fn
            )
            self.mlp_extractor = nn.Sequential(*mlp_extractor)
            final_features_dim = mlp_extractor_net_arch[-1] + cnn_concat_size
        else:
            self.mlp_extractor = None
            final_features_dim = total_concat_size

        self.cnn_extractors = nn.ModuleDict(cnn_extractors)
        self.flatten_extractors = nn.ModuleDict(flatten_extractors)

        # Update the features dim manually
        self._features_dim = final_features_dim

    def forward(self, observations: TensorDict) -> th.Tensor:

        # encode image obs through cnn
        cnn_encoded_tensor_list = []
        for key, extractor in self.cnn_extractors.items():
            cnn_encoded_tensor_list.append(extractor(observations[key]))

        # flatten vector obs
        flatten_encoded_tensor_list = []
        for key, extractor in self.flatten_extractors.items():
            flatten_encoded_tensor_list.append(extractor(observations[key]))

        # encode combined flat vector obs through mlp extractor (if set)
        # and combine with cnn outputs
        if self.mlp_extractor is not None:
            extracted_tensor = self.mlp_extractor(th.cat(flatten_encoded_tensor_list, dim=1))
            comb_extracted_tensor = th.cat([*cnn_encoded_tensor_list, extracted_tensor], dim=1)
        else:
            comb_extracted_tensor = th.cat([*cnn_encoded_tensor_list, *flatten_encoded_tensor_list], dim=1)

        return comb_extracted_tensor
