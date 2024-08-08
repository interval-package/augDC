import torch
import gym

class AlgBase():
    @torch.no_grad()
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def save(self, model_path):
        state_dict = dict()
        for model_name, model in self.models.items():
            state_dict[model_name] = model.state_dict()
        torch.save(state_dict, model_path)

    def load(self, model_path):
        state_dict = torch.load(model_path, weights_only=True)
        for model_name, model in self.models.items():
            model.load_state_dict(state_dict[model_name])
    pass