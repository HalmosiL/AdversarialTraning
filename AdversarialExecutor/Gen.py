from Adversarial import model_immer_attack_auto_loss
import torch

def run(id_, batch, device, model, attack, number_of_steps, data_queue):
    print("Gen_", id_, " started..")

    image = batch[0].to(device)
    label = batch[1].to(device)

    image = model_immer_attack_auto_loss(
        image=image,
        model=model,
        attack=attack,
        number_of_steps=number_of_steps,
        device=device
    )

    torch.save(image, data_queue + 'image_' + str(id_) + '.pt')
    torch.save(label, data_queue + 'label_' + str(id_) + '.pt')