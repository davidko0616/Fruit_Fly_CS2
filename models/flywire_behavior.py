"""Multi-head human-control readout over the FlyWire connectome trunk."""
import torch
import torch.nn as nn


DIRECTION_NAMES = ('forward', 'backward', 'strafe_left', 'strafe_right')
BUTTON_NAMES = ('fire', 'walk', 'crouch', 'jump', 'secondary_fire')


class FlyWireBehaviorCloner(nn.Module):
    """Predict simultaneous key duties and yaw from motor-neuron activity."""

    def __init__(self, backbone, yaw_scale_degrees=90.0):
        super().__init__()
        if float(yaw_scale_degrees) <= 0:
            raise ValueError('yaw_scale_degrees must be positive')
        self.backbone = backbone
        self.yaw_scale_degrees = float(yaw_scale_degrees)
        motor_size = len(backbone.output_idx)
        self.direction_head = nn.Linear(motor_size, len(DIRECTION_NAMES))
        self.yaw_head = nn.Linear(motor_size, 1)
        self.button_head = nn.Linear(motor_size, len(BUTTON_NAMES))
        for parameter in self.backbone.output_proj.parameters():
            parameter.requires_grad_(False)

    def forward(self, observation):
        output, _ = self.forward_with_state(observation)
        return output

    def forward_with_state(self, observation, previous_state=None,
                           temporal_decay=1.0):
        motor, state = self.backbone.motor_features(
            observation, previous_state, temporal_decay)
        output = {
            'direction_logits': self.direction_head(motor),
            'yaw_normalized': torch.tanh(self.yaw_head(motor)).squeeze(-1),
            'button_logits': self.button_head(motor),
        }
        return output, state

    def predict_controls(self, observation):
        output = self(observation)
        return {
            'direction_duties': output['direction_logits'].sigmoid(),
            'yaw_delta_degrees': (output['yaw_normalized'] *
                                  self.yaw_scale_degrees),
            'button_duties': output['button_logits'].sigmoid(),
        }
