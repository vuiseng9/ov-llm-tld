import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from openvino.runtime import Core
from transformers import HfArgumentParser


@dataclass
class Args:
    xml_path: str
    min_weight_numel: int = 8


def main(args: Args):
    ie = Core()
    model = ie.read_model(model=args.xml_path)
    items = []

    for op in model.get_ordered_ops():
        if 'constant' in str(op.get_type_info()).lower():
            if np.prod(op.get_output_shape(0)) < args.min_weight_numel:
                continue
            sparsity = (op.get_vector() == 0).mean()
            items.append(dict(
                name=op.friendly_name,
                shape=op.get_output_shape(0),
                type=op.get_element_type(),
                sparsity=sparsity,
            ))
    df = pd.DataFrame(items)
    print(df.to_string())
    return df


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(HfArgumentParser(Args).parse_args_into_dataclasses()[0])
