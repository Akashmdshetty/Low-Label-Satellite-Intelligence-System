import torch

class EMA:
    """
    Exponential Moving Average (EMA) updater for BYOL.
    
    Why BYOL needs EMA:
    In BYOL, the target network provides the regression targets for the online 
    network. If the target network's weights were updated via gradient descent, 
    the system would quickly collapse to a trivial constant solution. EMA 
    provides a slowly evolving 'mean' of the online network's weights, which 
    serves as a stable target.
    
    How EMA stabilizes training:
    By using a high momentum (e.g., 0.996), the target network changes very 
    slowly. This creates a more consistent optimization landscape for the 
    online network. It effectively acts as a form of temporal smoothing over 
    the online network's parameters.
    
    How it prevents collapse:
    EMA, combined with the predictor head (asymmetry) and normalization, 
    ensures that the target network is always 'ahead' or 'different' from the 
    online network in a way that preserves information. The stop-gradient on 
    the target network is crucial; EMA allows the target to inherit knowledge 
    from the online network without being directly influenced by the 
    collapse-prone gradients of the Siamese structure.
    """
    def __init__(self, momentum=0.996):
        self.m = momentum

    @torch.no_grad()
    def update(self, target_model, online_model):
        """
        Updates target_model parameters as a moving average of online_model parameters.
        target = m * target + (1 - m) * online
        """
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            # We use lerp_ for efficient in-place linear interpolation:
            # target = target + (1-m) * (online - target) => target = m*target + (1-m)*online
            target_param.data.lerp_(online_param.data, 1.0 - self.m)
            
        # Note: We only update parameters, not buffers (like BatchNorm running stats)
        # as per official BYOL implementation where target network usually shares 
        # or evolves buffers differently, but the primary mechanism is parameter EMA.
