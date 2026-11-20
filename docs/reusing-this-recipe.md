# How to build the next single-Spark recipe

Use the same release steps for each checkpoint and engine pair. Each hardware claim belongs to the exact configuration and test that produced it.

1. **Qualify a candidate.** Record the model ID, weight revision and format, engine and build revision, hardware reported by the source, and upstream usage terms. Verify that the artifacts are downloadable on the target architecture. Use the [backlog](model-backlog.md) for leads and data gaps.
2. **Pin the build.** Lock the source commits, patch set, base image by registry digest, toolchain checksums, and weight revision. Put local model paths and the volatile build cache outside Git. Keep evidence of which inputs arrived from upstream or had to be staged manually.
3. **Define the serving contract.** Expose plan, preflight, preparation, start, check, and stop commands. Start on loopback, use a dedicated container and image tag, and check for port and ownership conflicts before stopping or replacing workloads.
4. **Prove correctness on one node.** Verify the server’s reported model, bound socket, text output, and each supported modality with synthetic fixtures and known answers. Run independently of any private chat, files, or services.
5. **Measure capacity and latency.** Record requested context, actual token counts, successful concurrency, first content latency, engine decode stats when available, all-stream throughput, and host memory at named checkpoints. Separate empty-cache installation, cold serve, warmed requests, repeated-prefix queries, and multi-stream runs.
6. **Publish measured scope.** Include a run record with artifact identifiers, tests executed, failures, and known gaps. Use labels such as upstream-reported, booted locally, smoke-tested locally, benchmarked locally, and independently reproduced. Keep external measurements attributed.
7. **Review the public payload.** Scan all staged files and Git history. Keep tokens, SSH keys, machine identities, live logs, and private paths off the public branch. Link the source and notes for anything fetched at build time.

For a later skill, use the lifecycle commands as its interface and this checklist as the acceptance criteria. Preserve prompts, expected results, and output schemas in the recipe so agents and human contributors can run the same checks.
