## Experimental Deforum Fork

This is an experimental fork of the Deforum A1111 extension with the goal of refactoring the core and allow for direct control of "turbo-frames".
It will eventually also require an experimental version of Parseq to supply new "is_turbo_frame" values, future fork here: https://github.com/Tok/sd-parseq

**Current Status:**
This is a work in progress, and installation is not recommended yet.
Please refer to the original project for a stable version: [https://github.com/deforum-art/sd-webui-deforum](https://github.com/deforum-art/sd-webui-deforum)

## Neo-Core

This section details the changes made to the render core in this fork.

For easy integration, this fork isolates changes to the `render.py` module and introduces the `rendering` package.
Existing code in all other Deforum modules remains untouched.

* **Focus on Maintainability:** The core rendering functionality is being refactored step-by-step with a focus on improved readability, testability, and easier future modifications.
* **Key Improvements:**
  * Reduced cyclomatic complexity of the `render_animation` method.
  * Improved separation of concerns (e.g., dedicated module for printing).
  * Reduced argument complexity and improved scope control and variable organization using dataclasses.
  * Enhanced code clarity with improved naming and removal of obsolete comments.
    * But preserving domain specific lingo.
  * Improved unit testing capabilities due to a more modular structure and due to using expressions in place of statements where applicable.

**New Rendering Modules:**
* `rendering/data`: Provides a set of dataclasses specifically designed for data and logic handling within `render.py`.
* `rendering/util`: Contains stateless utility functions for data transformation and manipulation used in `render.py`.
* `rendering/util/call`: Provides modules to forward calls to other Deforum modules and adapt them to work with the new data structures without modifying the original code and without polluting any receiver namespace.

**Implementation Details:**
* **Multiparadigmatic:** The code leverages a procedural core with functional tools to transform object-oriented data structures.
* **Style and Standards:** The code adheres to PEP 8 style guidelines and to other such practices.
