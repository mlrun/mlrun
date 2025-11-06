# Cleanup K3S Action

A reusable GitHub composite action that checks for and cleans up any existing K3S installation.

## Description

This action:
- Detects if K3S is installed or running on the system
- Stops K3S services gracefully
- Kills any remaining K3S processes
- Runs the K3S uninstaller if available
- Cleans up K3S-related files and directories
- Optionally cleans up Docker containers and images

## Usage

```yaml
- name: Cleanup K3S installation
  uses: ./.github/actions/cleanup-k3s
```

### With conditional execution

```yaml
- name: Cleanup K3S installation
  if: always()
  uses: ./.github/actions/cleanup-k3s
```

## What it does

1. **Detection**: Checks for K3S installation by looking for:
   - `/usr/local/bin/k3s` binary
   - `/usr/local/bin/k3s-uninstall.sh` script
   - Running K3S processes

2. **Service Cleanup**: Stops systemd services:
   - `k3s.service`
   - `k3s-agent.service`

3. **Process Cleanup**: Kills any remaining K3S processes

4. **File Cleanup**: Removes K3S-related files and directories:
   - `/etc/rancher`
   - `/var/lib/rancher`
   - `/var/lib/kubelet`
   - `/var/lib/cni`
   - `/run/k3s`
   - K3S systemd services
   - K3S binaries

5. **Docker Cleanup** (optional): Prunes Docker system if Docker is running

## Requirements

- Linux system with systemd
- sudo access for cleanup operations
- bash shell

## Error Handling

All cleanup operations use `|| true` to ensure the action continues even if individual steps fail. This makes it safe to use in both setup and teardown phases.

## License

Apache License 2.0 - See LICENSE file for details
