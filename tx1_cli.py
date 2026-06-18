# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
from pathlib import Path
from typing import Annotated, List, Optional

import typer
from omegaconf import OmegaConf as om

# define Typer app
app = typer.Typer(help="Tx1 command line interface.")


# helper function to apply configuration overrides
def _apply_overrides(cfg, overrides):
    if not overrides:
        return cfg
    changes = {}
    for item in overrides:
        if "=" not in item:
            raise typer.BadParameter(f"Override must be key=value, got: {item}")
        key, val = item.split("=", 1)
        try:
            parsed = om.create(val)
        except Exception:
            parsed = val
        om.update(changes, key, parsed, merge=True)
    return om.merge(cfg, changes)


# root command
@app.callback(invoke_without_command=True)
def root(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        typer.echo("Tx1 command line interface.")
        typer.echo("Use `tx1 emb` to extract embeddings from an AnnData (.h5ad) file.")
        typer.echo("Run `tx1 emb --help` for detailed options.")


# emb command
@app.command("emb")
def emb(
    f: Annotated[
        Optional[Path],
        typer.Option(
            "-f",
            "--config",
            help="Path to YAML config.",
        ),
    ] = None,
    set_: Annotated[
        Optional[List[str]],
        typer.Option(
            "--set",
            help="Override config values (repeatable), e.g. --set paths.adata_input=data.h5ad",
        ),
    ] = None,
    hf_repo: Annotated[
        Optional[str],
        typer.Option(
            "--hf-repo",
            help="HF repo id, e.g. tahoebio/Tahoe-x1",
        ),
    ] = None,
    model_size: Annotated[
        Optional[str],
        typer.Option(
            "--model-size",
            help="Model size: 70m, 1b, or 3b",
        ),
    ] = None,
    input: Annotated[
        Optional[Path],
        typer.Option(
            "--input",
            "-i",
            help="Input .h5ad",
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output .h5ad",
        ),
    ] = None,
    gene_id_key: Annotated[
        Optional[str],
        typer.Option(
            "--gene-id-key",
            help="Column containing gene IDs for input adata.var",
        ),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option(
            "--batch-size",
            help="Batch size for inference (default: 64)",
        ),
    ] = None,
    seq_len: Annotated[
        Optional[int],
        typer.Option(
            "--seq-len",
            help="Sequence length to use during inference (default: 2048)",
        ),
    ] = None,
    return_gene_embeddings: Annotated[
        bool,
        typer.Option(
            "--return-gene-embs",
            help="Also compute gene embeddings",
        ),
    ] = False,
):

    # build configuration
    if f:
        cfg = om.load(str(f))
        cfg = _apply_overrides(cfg, set_ or [])
    else:
        if not (hf_repo and model_size and input and output and gene_id_key):
            raise typer.BadParameter(
                "Provide either -f CONFIG.yaml or flags: --hf-repo, --model-size, --input, --output, --gene-id-key (and optional others).",
            )
        cfg = om.create(
            {
                "model_name": "tx1",
                "paths": {
                    "hf_repo_id": hf_repo,
                    "hf_model_size": model_size,
                    "adata_input": str(input),
                    "adata_output": str(output),
                },
                "data": {
                    "gene_id_key": gene_id_key,
                },
                "predict": {
                    "batch_size": batch_size or 64,
                    "seq_len_dataset": seq_len if seq_len is not None else 2048,
                    "return_gene_embeddings": bool(return_gene_embeddings),
                },
            },
        )

    # run prediction
    from tahoe_x1.inference import predict_embeddings

    _ = predict_embeddings(cfg)
    typer.echo("Done.")


# main function
def main():
    app()


# entry point
if __name__ == "__main__":
    main()
