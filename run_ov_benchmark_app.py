from pathlib import Path
from transformers import HfArgumentParser
from dataclasses import dataclass
from typing import cast
import json
import subprocess
import tempfile


@dataclass
class Args:
    xml_path: str = 'openvino_model.xml'
    save_path: str = './ov_benchmarkapp/debug/'
    sparse_rate: float = None  # set it to 0~1 to enable
    add_bind: bool = False
    infer_precision: str = 'f32'
    use_data_shape: bool = True
    inference_num_threads: int = None

    n_iter: int = None
    time: int = None

    ctx_len: int = 511
    num_trials: int = 1

    def __post_init__(self):
        assert self.n_iter is None or self.time is None
        if self.sparse_rate is None:
            name = Path(self.xml_path).parent.name
            if 'sparse' in name:
                model_sparsity = int(name.split('sparse')[-1])
                self.sparse_rate = (model_sparsity - 10) / 100.0
            else:
                self.sparse_rate = -1
            print('Auto setting sparse_rate =', self.sparse_rate)


def get_shape_with_kvcache(batch_size, ctxlen):
    # 2nd token for mpt
    hidden_size, num_attention_heads, num_hidden_layers = 4096, 32, 32
    shape_str = f'input_ids[{batch_size},1],attention_mask[{batch_size},{ctxlen+1}]'
    for i in range(num_hidden_layers):
        kv_shape = f"[{batch_size},{num_attention_heads},{ctxlen},{hidden_size//num_attention_heads}]"
        shape_str += f",past_key_values.{i}.value{kv_shape}"
        shape_str += f",past_key_values.{i}.key{kv_shape}"
    return shape_str


def get_sparse_rate_cfg_path(sparse_rate: float, inference_num_threads, tmp_folder: str):
    if not (0 <= sparse_rate <= 1 or (inference_num_threads is not None)):
        return None
    cfg_path = Path(tmp_folder, 'cfg.json')
    cfg = {
        "CPU": {},
    }
    if 0 <= sparse_rate <= 1:
        cfg['CPU']["CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE"] = sparse_rate
    if inference_num_threads is not None:
        cfg['CPU']["INFERENCE_NUM_THREADS"] = inference_num_threads
        cfg['CPU']['NUM_STREAMS'] = 1
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
    return cfg_path


def run_process(cmd):
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        p.wait()
        stdout = p.stdout.read().decode()
        stderr = p.stderr.read().decode()
        return stdout, stderr


def main():
    args = HfArgumentParser(Args).parse_args_into_dataclasses()[0]
    args = cast(Args, args)
    print(args)
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    # data shape
    shape_key = '-data_shape' if args.use_data_shape else '-shape'
    shape = get_shape_with_kvcache(batch_size=1, ctxlen=args.ctx_len)

    # tld feature $ inference_num_threads
    sparse_rate_cfg = ''
    cfg_path = get_sparse_rate_cfg_path(
        args.sparse_rate, args.inference_num_threads, args.save_path)
    if cfg_path is not None:
        sparse_rate_cfg = ' -load_config ' + cfg_path.absolute().as_posix()

    # socket bind
    one_socket_bind = ''
    if args.add_bind:
        one_socket_bind = 'numactl --cpunodebind 0 --membind 0 '

    # timing
    if args.n_iter is not None:
        timing_str = f'-niter {args.n_iter}'
    else:
        timing_str = f'-t {args.time}'

    cmd = " ".join(map(str, [
        one_socket_bind,
        'benchmark_app',
        "-m", args.xml_path,
        shape_key, shape,
        timing_str,
        '-hint', 'latency',
        sparse_rate_cfg,
        "-infer_precision", args.infer_precision,
    ]))
    with open(Path(args.save_path, f'cmd.txt'), 'w', encoding='utf-8') as f:
        f.write(cmd)

    for i in range(args.num_trials):
        stdout, stderr = run_process(cmd)
        with open(Path(args.save_path, f'cmd_stdout_{i:02d}.log'), 'w', encoding='utf-8') as f:
            f.write(f'[stderr]\n{stderr}\n\n[stdout]\n{stdout}')
        if len(stderr) > 10:
            print('ERROR!')


main()
