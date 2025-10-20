(security)=
# Security

This guide covers security configurations and considerations for MLRun deployments.

## Non-root user support

By default, MLRun assigns the root user to MLRun runtimes and pods. You can improve the security context by changing the security mode, which is implemented by Iguazio during installation, and applied system-wide:

- **Override**: Use the user id of the user that triggered the current run or use the nogroupid for group id. Requires Iguazio v3.5.1.
- **Disabled**: Security context is not auto applied (the system applies the root user). (default)

