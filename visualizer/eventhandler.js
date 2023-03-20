let canvasmoving = false;
let canvasmoved = false;

function GenerateDocument()
{
    for(var i = 0; i < 8; ++i)
    {
        document.getElementById("board").innerHTML += 
        `<div style="padding-top:10px">
            <span style="width:40px; display:inline-block">
            #${i}
            </span><!--
            --><input class="latentslider" id="slider_l${i}" type="range" min="-400" max="400" value="0">
            <span id="value_l${i}">
                0
            </span>
            <div class="latentbox">
                <div class="latentinnerbox" id="box_l${i}"></div>
            </div>
        </div>`
    }
}

function AssignEventHandlers()
{
    canvas = document.getElementById("canvas");
    canvas.addEventListener("mousedown", OnCanvasMouseDown);
    canvas.addEventListener("mouseup", OnCanvasMouseUp);
    canvas.addEventListener("blur", OnCanvasMouseUp);
    canvas.addEventListener("mousemove", OnCanvasMouseMove);
    canvas.addEventListener("wheel", OnCanvasWheel);

    document.getElementById("slider_step").addEventListener("input", OnStepSliderChanged)
    document.getElementById("textbox_step").addEventListener("change", OnStepTextChanged)
    document.getElementById("checkbox_blur").addEventListener("change", (event)=>{draw_potential = document.getElementById("checkbox_blur").checked; DrawCanvas();})
    document.getElementById("button_d100").addEventListener("click", (event)=>{document.getElementById("textbox_step").value = (Number(current_step) - 100); OnStepTextChanged();})
    document.getElementById("button_d20").addEventListener("click", (event)=>{document.getElementById("textbox_step").value = (Number(current_step) - 20); OnStepTextChanged();})
    document.getElementById("button_i20").addEventListener("click", (event)=>{document.getElementById("textbox_step").value = (Number(current_step) + 20); OnStepTextChanged();})
    document.getElementById("button_i100").addEventListener("click", (event)=>{document.getElementById("textbox_step").value = (Number(current_step) + 100); OnStepTextChanged();})

    for(var i = 0; i < 8; ++i)
        document.getElementById("slider_l" + i).addEventListener("input", ((a)=>{return function(event){OnLatentSliderChanged(a);};})(i) )
}

function OnCanvasMouseDown(event)
{
    canvasmoving = true;
    canvasmoved = false;
}
function OnCanvasMouseUp(event)
{
    canvasmoving = false;
    if(!canvasmoved)
    {
        CanvasClick(event.clientX, event.clientY);
    }
}

function OnCanvasMouseMove(event)
{
    if(canvasmoving)
    {
        canvasmoved = true;
        viewport_x += event.movementX * viewport_screen_ratio;
        viewport_y += event.movementY * viewport_screen_ratio;
        DrawCanvas();
    }
}
function OnCanvasWheel(event)
{
    if(event.deltaY < 0)
    {
        viewport_scale *= 1.5;
        viewport_x = (viewport_x - half_viewport_width) * 1.5 + half_viewport_width;
        viewport_y = (viewport_y - half_viewport_height) * 1.5 + half_viewport_height;
    }
    else if(event.deltaY > 0)
    {
        viewport_scale /= 1.5;
        viewport_x = (viewport_x - half_viewport_width) / 1.5 + half_viewport_width;
        viewport_y = (viewport_y - half_viewport_height) / 1.5 + half_viewport_height;
    }
    DrawCanvas();
}
function OnStepTextChanged(event)
{
    current_step = document.getElementById("textbox_step").value;
    document.getElementById("slider_step").value = current_step;
    RequestCurrentMap();
}
function OnStepSliderChanged(event)
{
    current_step = document.getElementById("slider_step").value;
    document.getElementById("textbox_step").value = current_step;
    RequestCurrentMap();
}

function OnLatentSliderChanged(i)
{
    console.log(i)
    document.getElementById("value_l" + i).innerHTML = document.getElementById("slider_l" + i).value / 100
    if(clicked != -1)
    {
        RequestLatentResult();

    }
}