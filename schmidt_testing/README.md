# Schmidt Cluster Setup

This folder contains a small Slurm test job for checking that your Schmidt Sciences / Parallel Works cluster account can submit work to the CS321M H100 GPU partition.

## 1. Create your account

Open the Schmidt Sciences Parallel Works portal:

```text
https://schmidtsciences.parallel.works/
```

During first login:

1. Read and accept the legal terms.
2. Click **Forgot password?**
3. Use the same Stanford email address that received the cluster notice.
4. Follow the password reset email.
5. Configure two-factor authentication.

## 2. Install the Parallel Works CLI

On your local machine:

```bash
curl -fsSL https://activate.parallel.works/cli/install.sh | bash
```

Check that the CLI is available:

```bash
pw --help
```

If your shell cannot find `pw`, add the install directory to your `PATH`. The installer may place it at:

```bash
~/.local/bin/pw
```

## 3. Authenticate the CLI

The `pw auth login` command may only print help. Use an API key instead:

```bash
pw auth apikey
```

Paste your Parallel Works API key when prompted.

Confirm authentication:

```bash
pw auth whoami
```

## 4. SSH into the Schmidt cluster

From your local terminal:

```bash
pw ssh schmidt
```

After connecting, you should see a shell prompt on the cluster. For example:

```bash
workspace:~$
```

Check the available Slurm partitions:

```bash
sinfo
```

For CS321M, look for:

```text
cs321m
```

## 5. Put the test script on the cluster

Files on your laptop do not automatically exist on the cluster. If `sbatch` says it cannot open the file, create the file on the cluster.

On the cluster:

```bash
mkdir -p ~/schmidt_testing
cat > ~/schmidt_testing/MyFirstTestRun.sh <<'EOF'
#!/bin/bash
#SBATCH --job-name=MyFirstTestRun
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --partition=cs321m
#SBATCH --qos=cs321m
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

srun nvidia-smi -q
EOF
```

Note: `nano` may not be installed on the cluster, so the `cat <<'EOF'` method is the safest way to create the script.

## 6. Submit the test job

On the cluster:

```bash
sbatch ~/schmidt_testing/MyFirstTestRun.sh
```

If you see this error:

```text
sbatch: error: Batch job submission failed: Invalid qos specification
```

submit with the QoS flag:

```bash
sbatch --qos=cs321m ~/schmidt_testing/MyFirstTestRun.sh
```

The working script should include both:

```bash
#SBATCH --partition=cs321m
#SBATCH --qos=cs321m
```

## 7. Check the job

Show your queued or running jobs:

```bash
squeue -u $USER
```

After the job finishes, inspect the output:

```bash
ls
cat MyFirstTestRun-*.out
cat MyFirstTestRun-*.err
```

If the output includes NVIDIA GPU information, your cluster access and GPU job submission are working.

## Useful commands

```bash
sinfo                 # show partitions and node state
squeue -u $USER       # show your jobs
sbatch script.sh      # submit a Slurm job
scancel <job_id>      # cancel a job
pwd                   # show current directory on the cluster
ls                    # list files in current directory
```

## Common issues

### `failed to create SDK client`

If `pw ssh schmidt` says there is no configured context, authenticate first:

```bash
pw auth apikey
```

### `Unable to open file`

The script path does not exist on the cluster. Create the file after running:

```bash
pw ssh schmidt
```

### `nano: command not found`

Use the `cat > file <<'EOF'` approach shown above, or use another editor that exists on the cluster.

### `Invalid qos specification`

Use the `cs321m` QoS:

```bash
sbatch --qos=cs321m ~/schmidt_testing/MyFirstTestRun.sh
```

or add this to the script:

```bash
#SBATCH --qos=cs321m
```
