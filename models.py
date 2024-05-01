# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


class SimpleHead(nn.Module):
    """backbone + projection head"""
    def __init__(self, in_dim, out_dim=128, head='mlp'):
        super(SimpleHead, self).__init__()
        if head == 'linear':
            self.head = nn.Linear(in_dim, out_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, out_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        return nn.functional.normalize(self.head(x), dim=1)
