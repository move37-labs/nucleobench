"""A set-like data structure with Time-To-Live (TTL) functionality."""

import collections
import xxhash


class TTLCounterSet(object):
    """
    A set-like data structure where elements have a Time-To-Live (TTL)
    controlled by a global counter.

    Elements are removed when the difference between the current global
    counter and the counter value at the time of addition exceeds the TTL.
    Removal occurs when the global counter is incremented.

    Adding an element that already exists will raise a ValueError.

    Attributes:
        ttl (int): The Time-To-Live value. Elements older than this
                   (in terms of global counter increments) are considered expired.
        global_counter (int): The current value of the global counter.
                              Starts at 0 and only increments.
        _elements (dict): A dictionary mapping elements to the global counter
                          value at the time they were added.
        _expirations (collections.OrderedDict): An ordered dictionary mapping
                                                expiration counter values to sets
                                                of elements that expire at that
                                                counter value. This helps in
                                                efficiently removing expired items.
    """

    def __init__(self, ttl: int, use_hashing: bool = True):
        """
        Initializes the TTLCounterSet.

        Args:
            ttl (int): The Time-To-Live for elements. Must be a non-negative integer.
        """
        if not isinstance(ttl, int) or ttl < 0:
            raise ValueError("TTL must be a non-negative integer.")
        self.ttl = ttl
        self.global_counter = 0
        self._elements = {}  # Stores element -> addition_counter_value
        # Stores expiration_counter_value -> set of elements expiring then
        self._expirations = collections.OrderedDict()
        
        self.use_hashing = use_hashing

    def _str2hash(self, element: str) -> int:
        """
        Converts a string element to a hash value.

        Args:
            element (str): The string to be hashed.

        Returns:
            int: The hash value of the string.
        """
        return xxhash.xxh64(element).intdigest()

    def add(self, element: str) -> None:
        """
        Adds an element to the set.

        If the element already exists in the set, a ValueError is raised.

        Complexity:
            - Average: O(1) for dictionary and set operations if element is new.
            - If element exists: O(1) for the check before raising an error.
        """
        if self.use_hashing:
            element = self._str2hash(element)
        if element in self._elements:
            raise ValueError(f"Element '{element}' already exists in the set.")

        # Add the element with the current global counter value
        self._elements[element] = self.global_counter
        expiration_time = self.global_counter + self.ttl + 1 # Element expires when counter becomes this value

        if expiration_time not in self._expirations:
            self._expirations[expiration_time] = set()
        self._expirations[expiration_time].add(element)
        # Ensure _expirations remains sorted by key. collections.OrderedDict
        # maintains insertion order. Since global_counter only increments,
        # new expiration_times will generally be non-decreasing, keeping it naturally sorted.

    def __contains__(self, element: str) -> bool:
        """
        Checks if an string is in the set.
        Does not consider its TTL status here, only its presence.
        Expired elements are removed during 'increment_counter'.

        Complexity: O(1) average (dictionary lookup).
        """
        if self.use_hashing:
            element = self._str2hash(element)
        return self._str2hash(element) in self._elements

    def increment_counter(self) -> None:
        """
        Increments the global counter and removes any elements that have expired.

        Complexity: O(M) where M is the number of elements expiring at the
                    current counter value(s). In the best case (no expirations),
                    it's O(1) (amortized if using popitem).
        """
        self.global_counter += 1

        # Remove expired items
        # Elements expire when global_counter > addition_counter + ttl,
        # which means they are removed when global_counter == addition_counter + ttl + 1.
        # So, items whose 'expiration_time' (calculated as addition_counter + ttl + 1)
        # is less than or equal to the new global_counter are expired.
        while self._expirations:
            # Get the earliest expiration time and its associated elements
            earliest_expiration_time, elements_to_expire = self._expirations.popitem(last=False)

            if earliest_expiration_time <= self.global_counter:
                # These elements are now expired
                for elem in elements_to_expire:
                    # Since re-adds are disallowed, if an element is in an expiration
                    # queue, its entry in _elements must correspond to that original addition.
                    # We remove it from _elements.
                    if elem in self._elements: # This check is a safeguard
                        del self._elements[elem]
                    # If elem was not in self._elements, it implies an inconsistency,
                    # which shouldn't happen with the current logic.
            else:
                # This expiration time (and all subsequent ones) is in the future.
                # Put it back into _expirations at the beginning.
                self._expirations[earliest_expiration_time] = elements_to_expire
                self._expirations.move_to_end(earliest_expiration_time, last=False)
                break # Stop checking further expiration times

    def __repr__(self) -> str:
        return (f"TTLCounterSet(ttl={self.ttl}, global_counter={self.global_counter}, "
                f"elements={{elem: add_time for elem, add_time in self._elements.items()}})")

    def __len__(self) -> int:
        """Returns the number of elements currently in the set."""
        return len(self._elements)

    def get_current_elements(self) -> set:
        """
        Returns a Python set of the current elements.

        Complexity: O(N) where N is the number of elements.
        """
        return set(self._elements.keys())