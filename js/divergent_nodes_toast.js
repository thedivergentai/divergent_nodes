import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "DivergentNodes.Toast",
    async setup() {
        // Listen for custom messages from the ComfyUI backend
        // The 'app.ws.on' method allows listening to WebSocket messages.
        // We'll define a custom message type for our toasts.
        app.ws.on("divergent_nodes_toast", (event) => {
            const { severity, summary, detail, life } = event.data;
            app.ui.dialog.show(detail); // Fallback to dialog for now, toast API is app.extensionManager.toast.add
            // The actual Toast API is app.extensionManager.toast.add
            // Need to ensure app.extensionManager.toast is available.
            // If not, we might need to wait for it or use a fallback.

            if (app.extensionManager && app.extensionManager.toast) {
                app.extensionManager.toast.add({
                    severity: severity || "info",
                    summary: summary || "Divergent Nodes",
                    detail: detail,
                    life: life || 3000
                });
            } else {
                // Fallback if toast API is not available (e.g., older ComfyUI version or extension not loaded)
                console.log(`Divergent Nodes Toast (Fallback): [${severity}] ${summary}: ${detail}`);
                alert(`Divergent Nodes: ${summary}\n${detail}`);
            }
        });

        console.log("Divergent Nodes Toast Extension Loaded.");
    },
});
