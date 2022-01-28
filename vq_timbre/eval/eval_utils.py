

def load_audio_and_reconstruct(model, dataloader, return_numpy=True):
    audio = iter(dataloader).next()
    audio_pred, _, _, _, _ = model(audio)

    if return_numpy:
        return audio.squeeze(0).numpy(), audio_pred.squeeze(0).detach().numpy()
    else:
        return audio, audio_pred