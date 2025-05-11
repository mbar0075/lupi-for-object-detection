<h1> Learning Using Privileged Information for Object Detection </h1>

<p align="right" style="text-align: right;">
  <strong>"Improves accuracy with no added complexity. Works with any detection model."</strong>
</p>
<br>
<p align="left" style="text-align: left;">
  <strong>"No extra parameters. Just smarter training with what you already use."</strong>
</p>


<p align='center'>
  <img src="Assets/Diagrams/Architecture LUPIv2.png" alt="Architecture" width="100%" height="auto">
</p>
 
 <h2> Abstract </h2>
<p align='justify'>
<i> Object detection is widely recognised as a foundational task within the field of computer vision, with applications spanning automation, medical imaging, and surveillance. Although numerous models and methods have been developed, attaining high detection accuracy often requires the utilisation of complex model architectures, especially those based on transformers. These models typically demand extensive computational resources for inference and large-scale annotated datasets for training, both of which contribute to the overall difficulty of the task.

To address these challenges, this work introduces a novel methodology incorporating the Learning Using Privileged Information (LUPI) paradigm within the object detection domain. The proposed approach is compatible with any object detection architecture and operates by introducing privileged information to a teacher model during training. This information is then distilled into a student model, resulting in more robust learning and improved generalisation without increasing the number of model parameters and complexity.

The methodology is evaluated on both general-purpose object detection tasks and a focused case study involving litter detection in visually complex, highly variable outdoor environments. These scenarios are especially challenging due to the small size and inconsistent appearance of target objects. Evaluation is conducted both within individual datasets and across multiple datasets to assess consistency and generalisation. A total of 120 models are trained, covering five well-established object detection architectures. Four datasets are used in the evaluation: three focused on UAV-based litter detection and one drawn from the Pascal VOC 2012 benchmark to assess performance in multi-label detection and generalisation.

Experimental results consistently demonstrate improvements in detection accuracy across all model types and dataset conditions when employing the LUPI framework. In nearly all cases, these performance boosts are achieved without increasing the number of parameters or altering the model architecture, confirming the viability of the proposed methodology as a lightweight and effective modification to existing object detection systems.
 </i>
</p>

<h2> Methodology </h2>
<p align='justify'>
The proposed methodology adopts the Learning Using Privileged Information (LUPI) paradigm, which utilises additional supervision during training to improve model performance. In this approach, privileged information is provided to a teacher model and subsequently distilled into a student model. The core steps are outlined below:

1. **Generating Privileged Information:**
For each image in the dataset, a single-channel bounding box mask is generated to serve as additional supervisory input.

2. **Training the Teacher Model:**
The teacher model is trained using both the original dataset and the privileged information. It receives multi-channel input and is optimised to predict object classes as well as the corresponding bounding box masks.

3. **Distilling Knowledge to the Student Model:**
During training, the student model learns from the soft labels produced by the teacher. The training process incorporates a loss function based on the cosine distance between the latent feature representations at the final backbone layer of both models, guiding the student to align its internal representations with those of the teacher.

</p>

<h2>Detection Results</h2>

<h3>UAV-Based Litter Detection: Within-Dataset Evaluation</h3>

<h4>SODA: Small Objects at Different Altitudes (Low-Altitudes)</h4>
<p align='center'>
  <img src="Assets/figures/SODA 01m Dataset (Single-label).png" alt="SODA 01m" width="100%" height="auto">
</p>

<p align='center'>
<table align="center">
  <tr>
    <td align="center">
      <img src="Assets/figures/dataset_clusters/visual1.jpg" alt="Basline Results"  width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Assets/figures/dataset_clusters/visual1.jpg" alt="Best Student Results" width="100%" height="auto" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Baseline Results</b>
    </td>
    <td align="center">
      <b>Student Results (Ours)</b>
    </td>
</table>
</p>

<hr>

<h4>SODA: Small Objects at Different Altitudes (All-Altitudes)</h4>

<p align='center'>
  <img src="Assets/figures/SODA Dataset (Tiled Binary Detection).png" alt="SODA 01m" width="100%" height="auto">
</p>

<p align='center'>
<table align="center">
  <tr>
    <td align="center">
      <img src="Assets/figures/dataset_clusters/visual3.jpg" alt="Basline Results"  width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Assets/figures/dataset_clusters/visual4.jpg" alt="Best Student Results" width="100%" height="auto" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Baseline Results</b>
    </td>
    <td align="center">
      <b>Student Results (Ours)</b>
    </td>
</table>
</p>


<h3>UAV-Based Litter Detection: Across-Dataset Evaluation</h3>

<h4>BDW: Bottle Detection in the Wild Using Low-Altitude Unmanned Aerial Vehicles</h4>
<p align='center'>
  <img src="Assets/figures/BDW Dataset (Tested on Models Trained on SODA 01m Binary Detection).png" alt="SODA 01m" width="100%" height="auto">
</p>

<p align='center'>
<table align="center">
  <tr>
    <td align="center">
      <img src="Assets/figures/dataset_clusters/visual9.jpg" alt="Basline Results"  width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Assets/figures/dataset_clusters/visual10.jpg" alt="Best Student Results" width="100%" height="auto" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Baseline Results</b>
    </td>
    <td align="center">
      <b>Student Results (Ours)</b>
    </td>
</table>
</p>

<hr>

<h4>UAVVaste: Vision‐Based Trash and Litter Detection in Low Altitude Aerial Images</h4>

<p align='center'>
  <img src="Assets/figures/UAVVaste Dataset (Tested on Models Trained on SODA Tiled Binary Detection).png" alt="SODA 01m" width="100%" height="auto">
</p>

<p align='center'>
<table align="center">
  <tr>
    <td align="center">
      <img src="Assets/figures/dataset_clusters/visual11.jpg" alt="Basline Results"  width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Assets/figures/dataset_clusters/visual12.jpg" alt="Best Student Results" width="100%" height="auto" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Baseline Results</b>
    </td>
    <td align="center">
      <b>Student Results (Ours)</b>
    </td>
</table>
</p>

<h3>Multi-label Object Detection: Pascal VOC 2012 Evaluation</h3>

<p align='center'>
  <img src="Assets/figures/Pascal VOC 2012 Dataset (Multi-label Detection for 20 Classes) .png" alt="Pascal VOC" width="100%" height="auto">
</p>

<p align='center'>
  <img src="Assets/figures/map_all_classes_pascal_voc_no_text.png" alt="Pascal VOC CM" width="100%" height="auto">
</p>

<hr>

<p align='center'>
<table align="center">
  <tr>
    <td align="center">
      <img src="Assets/figures/dataset_clusters/visual5.jpg" alt="Basline Results"  width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Assets/figures/dataset_clusters/visual6.jpg" alt="Best Student Results" width="100%" height="auto" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Assets/figures/dataset_clusters/visual7.jpg" alt="Basline Results"  width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Assets/figures/dataset_clusters/visual8.jpg" alt="Best Student Results" width="100%" height="auto" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Assets/figures/dataset_clusters/visual13.jpg" alt="Basline Results"  width="100%" height="auto" />
    </td>
    <td align="center">
      <img src="Assets/figures/dataset_clusters/visual14.jpg" alt="Best Student Results" width="100%" height="auto" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Baseline Results</b>
    </td>
    <td align="center">
      <b>Student Results (Ours)</b>
    </td>
</table>
</p>

<!-- <h2>Citation</h2>

```bibtex
``` -->

<h2>Installation</h2>

```bash
git clone https://github.com/mbar0075/lupi-for-object-detection.git
cd lupi-for-object-detection
pip install -r requirements.txt
```