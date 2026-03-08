import os
import yaml

class Logger:
    """
    Unified Logger for the BYOL project.
    Supports console logging and optional Weights & Biases (WandB) integration.
    """
    def __init__(self, use_wandb=False, project_name="ssl-byol-eurosat", config=None):
        self.use_wandb = use_wandb
        
        if self.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(project=project_name, config=config)
                print(f"WandB initialized for project: {project_name}")
            except ImportError:
                print("WandB not installed. Falling back to console logging only.")
                self.use_wandb = False
        else:
            print("Console logging only (WandB disabled).")

    def log_metrics(self, metrics_dict, step):
        """
        Logs a dictionary of metrics at a specific step.
        Logs to console and WandB if enabled.
        """
        # Console output
        log_str = f"Step {step} | " + " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics_dict.items()])
        print(log_str)
        
        # WandB output
        if self.use_wandb:
            import wandb
            wandb.log(metrics_dict, step=step)

    def save_config(self, config_dict, path="config_logged.yaml"):
        """
        Saves the configuration dictionary to a YAML file.
        Useful for reproducibility and tracking.
        """
        with open(path, 'w') as f:
            yaml.dump(config_dict, f)
        
        if self.use_wandb:
            import wandb
            wandb.save(path)
        
        print(f"Configuration saved to {path}")

    def finish(self):
        """Ends the logging session."""
        if self.use_wandb:
            import wandb
            wandb.finish()
