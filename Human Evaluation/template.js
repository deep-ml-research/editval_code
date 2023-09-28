<!-- You must include this JavaScript file -->
<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

<!-- For the full list of available Crowd HTML Elements and their input/output documentation,
      please refer to https://docs.aws.amazon.com/sagemaker/latest/dg/sms-ui-template-reference.html -->

<!-- You must include crowd-form so that your task submits answers to MTurk -->
<crowd-form answer-format="flatten-objects">

  <crowd-instructions link-text="View instructions" link-type="button">
    <short-summary>
        <p>You must rate how good the right-hand side image applies the given prompt to the left-hand side image, and how good are the other aspects of the left-hand side image preserved. </p>
    </short-summary>

    <detailed-instructions>
      <p>The left image is the original one and the right image is supposed to be an edited image that applies a specific prompt to the original image.</p>
      <p>You have to separately rate these three factos:</p>
        <ul>
          <li>How well is the prompt's instruction applied to the image</li>
          <li>How well are other properties of the main object of the image preserved after the edit</li>
          <li>How well are other parts of the image (other than the ones related to the main object) preserved after the edit? (e.g., background, other objects, coloring) </li>
        </ul>
    
    </detailed-instructions>
    
    <positive-example>
        <div class="row" style="display:flex;">
          <div class="column" style="float:left;padding:5px;">
            <img src="https://editbench.s3.amazonaws.com/edited_images/textual_inversion/object_replacement/113168/113168_unedited.png" alt="Original Image" style="width:300px; height:300px">
          </div>
          <div class="column" style="float:left;padding:5px;">
            <img src="https://editbench.s3.amazonaws.com/edited_images/textual_inversion/object_replacement/113168/113168_object_replacement_cat.png" alt="Edited Image" style="width:300px; height:300px">
          </div>
        </div>
        
          <div>
            <p>The right image is supposed to add the prompt "Replace the dog with cat" to the left image.</p>
          </div>
          
          <div>
            <p>How well is the edit from the given prompt applied?</p>
            <crowd-radio-group>
        	    <crowd-radio-button disabled>Not applied</crowd-radio-button>
        	    <crowd-radio-button disabled>Minorly applied</crowd-radio-button>
        	    <crowd-radio-button disabled>Adequetly applied</crowd-radio-button>
        	    <crowd-radio-button disabled checked>Perfectly applied</crowd-radio-button>
        	</crowd-radio-group>
        	<p style="color:red;">The dog is successfully replaced by a cat. Therefore we can say the edit is perfectly applied. </p>
          </div>
          
          <div>
            <p>How well are the other properties (other than what the edit is targeting) of the main object (dog) preserved in the right image?</p>
            <crowd-radio-group>
        	    <crowd-radio-button disabled>Object is completely changed</crowd-radio-button>
        	    <crowd-radio-button disabled>Some parts are preserved</crowd-radio-button>
        	    <crowd-radio-button disabled checked>Most parts are preserved</crowd-radio-button>
        	    <crowd-radio-button disabled>Other properties of the object are perfectly preserved</crowd-radio-button>
        	</crowd-radio-group>
        	<p style="color:red;">The coloring and style of the cat are similar to the dog to some extent, however the position of the object has been changed.</p>
          </div>
          
          
          <div>
            <p>How well are other aspects of the left image (other than main object, and other than what the edit is targeting) preserved in the right image?</p>
            <crowd-radio-group>
        	    <crowd-radio-button disabled>Completely changed</crowd-radio-button>
        	    <crowd-radio-button disabled checked>Some parts are preserved</crowd-radio-button>
        	    <crowd-radio-button disabled>Most parts are preserved</crowd-radio-button>
        	    <crowd-radio-button disabled>Perfectly preserved</crowd-radio-button>
        	</crowd-radio-group>
        	<p style="color:red;">The general vibe of the background is kept and the pet is still sitting on an electronic device. However, many properties of the environment and the device have been changed and they're not the same anymore.</p>
          </div>
    </positive-example>
    <positive-example>
        <div class="row" style="display:flex;">
          <div class="column" style="float:left;padding:5px;">
            <img src="https://editbench.s3.amazonaws.com/edited_images/pix2pix/background/195671/195671_unedited.png" alt="Original Image" style="width:300px; height:300px">
          </div>
          <div class="column" style="float:left;padding:5px;">
            <img src="https://editbench.s3.amazonaws.com/edited_images/pix2pix/background/195671/195671_background_forest_1.5_9.5.png" alt="Edited Image" style="width:300px; height:300px">
          </div>
        </div>
        
          <div>
            <p>The right image is supposed to add the prompt "Change the background to forest" to the left image.</p>
          </div>
          
          <div>
            <p>How well is the edit from the given prompt applied?</p>
            <crowd-radio-group>
        	    <crowd-radio-button disabled checked>Not applied</crowd-radio-button>
        	    <crowd-radio-button disabled>Minorly applied</crowd-radio-button>
        	    <crowd-radio-button disabled>Adequetly applied</crowd-radio-button>
        	    <crowd-radio-button disabled>Perfectly applied</crowd-radio-button>
        	</crowd-radio-group>
        	<p style="color:red;">The edit has only changed the color of some parts of the image to green and has not changed any parts of the environment to forest.</p>
          </div>
          
          <div>
            <p>How well are the other properties (other than what the edit is targeting) of the main object (potted plant) preserved in the right image?</p>
            <crowd-radio-group>
        	    <crowd-radio-button disabled>Object is completely changed</crowd-radio-button>
        	    <crowd-radio-button disabled>Some parts are preserved</crowd-radio-button>
        	    <crowd-radio-button disabled checked>Most parts are preserved</crowd-radio-button>
        	    <crowd-radio-button disabled>Other properties of the object are perfectly preserved</crowd-radio-button>
        	</crowd-radio-group>
        	<p style="color:red;">The potted plant has not changed much, but its color has unnecessarily been changed to green.</p>
          </div>
          
          <div>
            <p>How well are other aspects of the left image (other than main object, and other than what the edit is targeting) preserved in the right image?</p>
            <crowd-radio-group>
        	    <crowd-radio-button disabled>Completely changed</crowd-radio-button>
        	    <crowd-radio-button disabled>Some parts are preserved</crowd-radio-button>
        	    <crowd-radio-button disabled checked>Most parts are preserved</crowd-radio-button>
        	    <crowd-radio-button disabled>Perfectly preserved</crowd-radio-button>
        	</crowd-radio-group>
        	<p style="color:red;">The background has mostly not been changed, so we can say that most parts are preserved. Note that if the background was actually changed to a forest, you shouldn't necessarily chose the "Completely changed" option, because the question is aksing for changes other than what the edit is targeting.</p>
          </div>
    </positive-example>
  </crowd-instructions>
  
  <div>
    <p>&nbsp; "${prompt}"</p>
    <p>&nbsp; We obtain RIGHT image by applying above text-instruction to LEFT image </p>
  </div>


  <div class="row" style="display:flex;">
      <div class="column" style="float:left;padding:5px;">
        <img src="${url_org}" alt="Original Image" style="width:300px; height:300px">
      </div>
      <div class="column" style="float:left;padding:5px;">
            <img src="${url_edit}" alt="Edited Image" style="width:300px; height:300px">
      </div>
  </div>
  
 
  <div>
    <p>&nbsp; How well is the editing from given text prompt applied?</p>
    <crowd-radio-group>
	    <crowd-radio-button name="q1_0" value="0">Not applied</crowd-radio-button>
	    <crowd-radio-button name="q1_1" value="1">Minorly applied</crowd-radio-button>
	    <crowd-radio-button name="q1_2" value="2">Adequately applied</crowd-radio-button>
	    <crowd-radio-button name="q1_3" value="3">Perfectly applied</crowd-radio-button>
	</crowd-radio-group>
  </div>
  
  <div>
    <p>&nbsp;How well are the other properties (other than intended) of the main object (${class_name}) preserved in the RIGHT image?</p>
    <crowd-radio-group>
	    <crowd-radio-button name="q2_0" value="0">Object is completely changed</crowd-radio-button>
	    <crowd-radio-button name="q2_1" value="1">Some parts are preserved</crowd-radio-button>
	    <crowd-radio-button name="q2_2" value="2">Most parts are preserved</crowd-radio-button>
	    <crowd-radio-button name="q2_3" value="3">Other properties of the object are perfectly preserved</crowd-radio-button>
	</crowd-radio-group>
  </div>
  
  
  <div>
    <p>&nbsp; How well are other aspects of LEFT image (other than main object, and the intended changes) preserved in the RIGHT image?</p>
    <crowd-radio-group>
	    <crowd-radio-button name="q3_0" value="0">Completely changed</crowd-radio-button>
	    <crowd-radio-button name="q3_1" value="1">Some parts are preserved</crowd-radio-button>
	    <crowd-radio-button name="q3_2" value="2">Most parts are preserved</crowd-radio-button>
	    <crowd-radio-button name="q3_3" value="3">Perfectly preserved</crowd-radio-button>
	</crowd-radio-group>
  </div>


</crowd-form>
