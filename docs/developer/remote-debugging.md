# Remote Debugging

### MLRun API

Prerequisites:

1. Open PyCharm, [create](https://www.jetbrains.com/help/pycharm/remote-debugging-with-product.html#create-remote-debug-config) a run debug configuration
    - Path mapping `/path/to/mlrun/mlrun=/mlrun`
    - Leave host to be "localhost"
    - Check "Redirect output to console"
    - Uncheck "Suspect after connect"

2. Install `ngrok` and run it to create a tunnel. (e.g. `ngrok tcp 40000`)
    - Use the tcp port from (1).

3. Once ngrok is running, save the public url from the output and use it to connect to the remote debug server.
    ```
    ...
    Forwarding                    tcp://7.tcp.eu.ngrok.io:13209 -> localhost:40000
    ```

4. Add envvars to MLRun's API deployment (or use `mlrun-override-env` configmap) with below envvars
   - `MLRUN_API_DEBUG_MODE` - Whether to enable debugging or not (e.g.: `enabled`)
   - `MLRUN_API_DEBUG_HOST` - The host returned by ngrok (e.g.: `7.tcp.eu.ngrok.io`)
   - `MLRUN_REMOTE_DEBUG_PORT` - The port returned by ngrok (e.g.: `13209`)

5. Start `Remote Debugging` run configuration created previously

6. Once the deployment is patched with new dev image, the new running pod will try connecting to your PyCharm server. 
    Once it is connected, you will see on your IDE console `Connected to pydev debugger (build ...)`.

7. You can now use breaking points while running a remote MLRun API instance.
