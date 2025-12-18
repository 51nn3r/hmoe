from typing import Tuple, Dict, Any


class ArgumentParsingStrategy:
    """
    A strategy class that helps parse and rearrange positional and keyword arguments
    for pseudo-layers during forward calls. It can optionally cache the positions of
    required arguments, so subsequent parsing can be faster or consistent.
    """

    _required_keys: Dict[str, Any]
    _use_cache: bool
    _cached_positions: Dict[str, int] | None

    def __init__(
            self,
            required_keys: Dict[str, Any],
            use_cache: bool = True,
    ):
        """
        Initializes the argument parsing strategy.

        :param required_keys: A dictionary of required argument keys. This structure
                              indicates which arguments are mandatory for the pseudo-layer.
        :param use_cache: A boolean flag indicating whether to enable caching of the
                          argument positions once detected.
        """
        self._required_keys = required_keys
        self._use_cache = use_cache
        self._cached_positions = None

    def parse(
            self,
            *args,
            **kwargs,
    ):
        """
        Public entry point for parsing args/kwargs. Internally delegates to _parse_args().

        :param args: Positional arguments passed into the pseudo-layer's call.
        :param kwargs: Keyword arguments passed into the pseudo-layer's call.
        :return: A tuple of (remaining positional arguments, updated keyword arguments).
        """
        return self._parse_args(args, kwargs)

    def _parse_args(
            self,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
    ) -> (Tuple[Any], Dict[str, Any]):
        """
        Internal method that processes the required keys. If caching is enabled and
        positions are already known, it reassigns those positions in kwargs directly.
        Otherwise, it attempts to map any leftover positional arguments to required keys.

        :param args: The original tuple of positional arguments.
        :param kwargs: The original dictionary of keyword arguments.
        :return: A tuple containing the remaining positional arguments (after
                 extracting required keys) and the final kwargs dictionary.
        """

        # If we're using cache and we already have the positions, reuse them
        if self._use_cache and self._cached_positions is not None:
            for key, idx in self._cached_positions.items():
                kwargs[key] = args[idx]

            # The leftover arguments come after all required positions
            return args[len(self._cached_positions):], kwargs

        # If we haven't cached yet (or use_cache is False), we do a fresh parse
        new_cached_positions = {} if self._use_cache else None
        remaining_args = list(args)

        # For each required key, if it's not in kwargs, we pop from remaining_args
        # and assign it. Also optionally record the position if caching is on.
        for key in self._required_keys:
            if key not in kwargs:
                if not remaining_args:
                    raise ValueError(f"Missing required argument: {key}")

                kwargs[key] = remaining_args.pop(0)
                if self._use_cache:
                    # Position = (original length of args) - how many remain - 1
                    new_cached_positions[key] = len(args) - len(remaining_args) - 1

        # Save the newly found positions so next parse can be faster
        if self._use_cache:
            self._cached_positions = new_cached_positions

        return remaining_args, kwargs
