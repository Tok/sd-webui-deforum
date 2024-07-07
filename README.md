# Experimental Deforum Fork
This is an experimental fork of the Deforum A1111 extension with a refactored render core that allows for 
direct control of "turbo-frames" from Parseq. 

### Current Status
The forked extension is work in progress and may not be stable.
Installation is recommended **only for the purpose of using the Parseq based key frame redistribution features**.

⚠️ Some Deforum features like hybrid video and coherence algorithms are currently not working and may need to be turned off.

For a feature complete and stable version, please refer to the original project at: [https://github.com/deforum-art/sd-webui-deforum](https://github.com/deforum-art/sd-webui-deforum)

#### Installation
Since the name of this extension is shared with the original, it requires a full removal of the regular Deforum extension from Automatic 1111.
After that it can be installed from "Extensions" using "Install from URL": https://github.com/Tok/sd-webui-deforum
The forked extension adds a new "Neo Core" tab in Deforum, with further information on how to properly use it.

💡 The rest of this Readme can be skipped if you're not interested in the code.

## 🏎️ Parseq Key Frame Distribution

###  Purpose & Features

* Parseq Precision: All Parseq-provided key frames are guaranteed to be included in the final output, ensuring the highest possible sync precision.
* Cadence Flexibility: Cadence settings can be set to exceptionally high values (e.g., 10, 20+) or can be ignored completely without losing Parseq synchronization, enabling fast generations with less diffusion steps.
* No Workarounds: This approach eliminates the need for tricky workarounds when using Parseq with high or ignored cadence settings.

### 📊 Key Index Distribution Modes

New key in 'deforum_settings.txt': "neocore_key_index_distribution"

#### 🕳️ PARSEQ_ONLY (without cadence)
[`KeyIndexDistribution.PARSEQ_ONLY`](https://github.com/Tok/sd-webui-deforum/blob/automatic1111-webui/scripts/deforum_helpers/rendering/data/step/key_index_distribution.py)
This distribution **completely ignores cadence** and is only diffusing key frames defined in Parseq (=frames that have a table entry in Parseq). 
All non-key frames are handled as if they would be cadence frames, although that term doesn't really apply when the key frames are not spaced out in fixed intervals.

#### 〰️ UNIFORM_WITH_PARSEQ (variable pseudo cadence)
[`KeyIndexDistribution.UNIFORM_WITH_PARSEQ`](https://github.com/Tok/sd-webui-deforum/blob/automatic1111-webui/scripts/deforum_helpers/rendering/data/step/key_index_distribution.py)
This mode ensures a uniform key frame distribution according to the specified cadence settings, while also prioritizing the inclusion of all key frames provided by Parseq.

Here's how it works:
1. The uniform key frame distribution is calculated based on the cadence settings, creating a set of non-cadence key frames.
2. The Parseq-provided key frames are then overlaid on top of this uniform distribution, replacing the closest non-cadence key frame.

The `UNIFORM_WITH_PARSEQ` distribution prioritizes Parseq key frames over the uniform cadence-based distribution, providing a balance between sync precision and generation speed.

In essence, it means that **all Parseq key frames**, will be guaranteed to not
be a cadence-frames. We gain Parseq precision at the tradeoff of cadence regularity.

##### Pseudo Cadence
`cadence` is still loosely observed in this mode, but since key frames are rearranged, the cadence settings should be understood as an average value.
In UNIFORM_WITH_PARSEQ mode, a cadence setting of "10" means "about 10".

## 📼 Video Examples

### Parseq Synchronized Music Video
This music video was **generated in less than only 92 minutes on a 4070**, directly at 60 FPS and with a resolution of 1280x720.
It contains 6694 frames of which only 913 were actually diffused, but in full sync with Parseq. 
Calculated pseudo cadence is 6694 / 913 = 7.33. More details in video description.

"neocore_key_index_distribution": "Parseq Only"

#### Mark Fell - multistability 2-b
[![Mark Fell - multistability 2-b](https://img.youtube.com/vi/K-9V5Tntck4/maxresdefault.jpg)](https://www.youtube.com/watch?v=K-9V5Tntck4 "Click to watch on YouTube")

## 🧹 Refactored Render Core
This section details the changes made to the render core in this fork.

For easy integration, this fork isolates changes mostly to the [`render.py`](https://github.com/Tok/sd-webui-deforum/blob/automatic1111-webui/scripts/deforum_helpers/render.py) module and introduces the [`rendering`](https://github.com/Tok/sd-webui-deforum/blob/automatic1111-webui/scripts/deforum_helpers/rendering) package.
Existing code in other Deforum modules, remains mostly unchanged (exception is UI and args related code).

![image](https://i.kym-cdn.com/photos/images/original/001/399/018/31f.gif)

* **🛠️ Focus on Maintainability:** The core rendering functionality is being refactored step-by-step with a focus on improved readability, testability, and easier future modifications.
* **☝️ Key Improvements:**
  * Reduced cyclomatic complexity of the [`render_animation`](https://github.com/Tok/sd-webui-deforum/blob/automatic1111-webui/scripts/deforum_helpers/render.py#L41) method.
  * Improved separation of concerns (e.g., dedicated module for printing).
  * Reduced argument complexity and improved scope control and variable organization using dataclasses.
  * Enhanced code clarity with improved naming and removal of obsolete comments.
    * But preserving domain specific lingo.
  * Improved unit testing capabilities due to a more modular structure and due to using expressions in place of statements where applicable.

**🆕 New Rendering Modules:**
* [`rendering/img_2_img_tubes`](https://github.com/Tok/sd-webui-deforum/blob/automatic1111-webui/scripts/deforum_helpers/rendering/img_2_img_tubes.py): Provides functions for conditionally processing images through various transformations.
* [`rendering/data`](https://github.com/Tok/sd-webui-deforum/blob/automatic1111-webui/scripts/deforum_helpers/rendering/data): Provides a set of dataclasses specifically designed for data and logic handling within `render.py`.
* [`rendering/util`](https://github.com/Tok/sd-webui-deforum/blob/automatic1111-webui/scripts/deforum_helpers/rendering/util): Contains stateless utility functions for data transformation and manipulation used in `render.py`.
* [`rendering/util/call`](https://github.com/Tok/sd-webui-deforum/blob/automatic1111-webui/scripts/deforum_helpers/rendering/util/call): Provides modules to forward calls to other Deforum modules and adapt them to work with the new data structures without modifying the original code and without polluting any receiver namespace.

**🔎 Implementation Details:**
* **Multiparadigmatic:** The code leverages a procedural core with functional tools to transform object-oriented data structures.
* **Style and Standards:** The code adheres to PEP 8 style guidelines and to other such practices.
