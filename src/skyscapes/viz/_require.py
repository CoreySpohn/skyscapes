"""Import guard for the optional eyepiece dependency."""


def eyepiece():
    """Import eyepiece, or raise with the install hint.

    Returns:
        The imported ``eyepiece`` module.

    Raises:
        ImportError: If eyepiece is not installed; the message names the
            ``skyscapes[viz]`` extra that provides it.
    """
    try:
        import eyepiece
    except ImportError:
        raise ImportError(
            "skyscapes.viz requires eyepiece: pip install 'skyscapes[viz]'"
        ) from None
    return eyepiece
