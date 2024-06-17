import torch


class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, mask_vector=None, special_vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        self.mask_vector = None
        self.special_vector = None
        self.vector = None
        if mask_vector is not None:
            self.mask_vector = mask_vector
        elif special_vector is not None:
            self.special_vector = special_vector
        elif vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():

                pretrained_state_dict = pretrained_checkpoint.state_dict()
                finetuned_state_dict = finetuned_checkpoint.state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def ret_mask_vector(self):
        cur_vec = {}
        with torch.no_grad():
            for key in self.vector:
                vec = self.vector[key].numpy()
                vec = vec !=0
                vec = vec.astype(int)
                cur_vec[key] = vec

            self.mask_vector = cur_vec
        Temp = TaskVector(mask_vector=cur_vec)


    # For printing the difference vector
    def get_print(self):
        cur_vec = {}
        for key in self.vector:
            cur_vec[key] = self.vector[key]
        return cur_vec

    # For getting the masked vector
    def get_mask_vec(self):
        cur_vec = {}
        for key in self.mask_vector:
            cur_vec[key] = self.mask_vector[key]
        return cur_vec

    # For getting the special vector
    def get_special_vec(self):
        cur_vec = {}
        for key in self.special_vector:
            cur_vec[key] = self.special_vector[key]
        return cur_vec

    # Finding the special vector
    def calc_Special_Vector(self,mask_vector,diff_vector):
        Special_Matrix = {}

        with torch.no_grad():

            for key in self.vector: 

                Special_Matrix[key] = diff_vector[key] * mask_vector[key]
            self.special_vector = Special_Matrix
        Temp = TaskVector(special_vector=Special_Matrix)

    
    def apply_special_matrix(self,pretrained_model, special_mat, scaling_coef):
        """Apply a special matrix to a pretrained model."""
        with torch.no_grad():

            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] - scaling_coef * special_mat[key]
        pretrained_model.load_state_dict(new_state_dict, strict=True)
        return pretrained_model


