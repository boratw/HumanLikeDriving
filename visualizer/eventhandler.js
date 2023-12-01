let canvasmoving = false;
let canvasmoved = false;
let linechart = null;
var latent_length=8;
let playevent = null;

function GenerateDocument()
{
    for(var i = 0; i < 8; ++i)
    {
        document.getElementById("board").innerHTML += 
        `<div id="div_l${i}" style="padding-top:10px">
            <span style="width:40px; display:inline-block">
            #${i < 4 ? 'G' + i : 'L' + i}
            </span><!--
            --><input class="latentslider" id="slider_l${i}" type="range" min="-500" max="500" value="0">
            <span id="value_l${i}">0</span>
            <div class="latentbox">
                <div class="latentinnerbox" id="box_l${i}"></div>
            </div>
        </div>`
    }
    
    linechart = new Chart(document.getElementById('chart'),
        {
          type: 'line',
          data: {},
          options: {
              animation: false,
              elements:{
                point:{
                    pointStyle:false
                }
              },
              scales: {
                  y: {
                      max: 1.5,
                      min: -1.5,
                      ticks: {
                          stepSize: 0.5
                      }
                  },
                  x: {
                    ticks: {
                        stepSize: 100
                    }

                  }
            
              }
          }
        })
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
    document.getElementById("slider_latentstart").addEventListener("input", OnLatentStartSliderChanged)
    document.getElementById("textbox_latentstart").addEventListener("change", OnLatentStartTextChanged)
    document.getElementById("slider_latentend").addEventListener("input", OnLatentEndSliderChanged)
    document.getElementById("textbox_latentend").addEventListener("change", OnLatentEndTextChanged)
    document.getElementById("checkbox_blur").addEventListener("change", (event)=>{draw_potential = document.getElementById("checkbox_blur").checked; DrawCanvas();})
    document.getElementById("button_d100").addEventListener("click", (event)=>{document.getElementById("textbox_step").value = (Number(current_step) - 100); OnStepTextChanged();})
    document.getElementById("button_d20").addEventListener("click", (event)=>{document.getElementById("textbox_step").value = (Number(current_step) - 20); OnStepTextChanged();})
    document.getElementById("button_i20").addEventListener("click", (event)=>{document.getElementById("textbox_step").value = (Number(current_step) + 20); OnStepTextChanged();})
    document.getElementById("button_i100").addEventListener("click", (event)=>{document.getElementById("textbox_step").value = (Number(current_step) + 100); OnStepTextChanged();})
    document.getElementById("button_play").addEventListener("click", OnPlay)
    document.getElementById("button_stop").addEventListener("click", OnStop)
    document.getElementById("button_drawall").addEventListener("click", OnDrawAll)

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
        CanvasClick(event.offsetX, event.offsetY);
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
    RequestCurrentStep();
}
function OnStepSliderChanged(event)
{
    current_step = document.getElementById("slider_step").value;
    document.getElementById("textbox_step").value = current_step;
    RequestCurrentStep();
}
function OnLatentStartTextChanged(event)
{
    current_step = document.getElementById("textbox_latentstart").value;
    document.getElementById("slider_latentstart").value = current_step;
    if(clicked != -1)
        RequestLatentPredicted(clicked);
}
function OnLatentStartSliderChanged(event)
{
    current_step = document.getElementById("slider_latentstart").value;
    document.getElementById("textbox_latentstart").value = current_step;
    if(clicked != -1)
        RequestLatentPredicted(clicked);
}
function OnLatentEndTextChanged(event)
{
    current_step = document.getElementById("textbox_latentend").value;
    document.getElementById("slider_latentend").value = current_step;
    if(clicked != -1)
        RequestLatentPredicted(clicked);
}
function OnLatentEndSliderChanged(event)
{
    current_step = document.getElementById("slider_latentend").value;
    document.getElementById("textbox_latentend").value = current_step;
    if(clicked != -1)
        RequestLatentPredicted(clicked);
}

function OnLatentSliderChanged(i)
{
    document.getElementById("value_l" + i).innerHTML = document.getElementById("slider_l" + i).value / 100
    if(clicked != -1)
    {
        RequestOutput(clicked);

    }
}

function OnPlay()
{
    if(playevent == null)
    {
        playevent = setInterval(PlayStep, 1000 )
    }
}
function OnStop()
{
    if(playevent != null)
    {
        clearInterval(playevent)
        playevent = null;
    }
}
function PlayStep()
{
    document.getElementById("textbox_step").value = (Number(current_step) + 5);
    document.getElementById("textbox_latentstart").value = 0;
    document.getElementById("slider_latentstart").value = 0;
    document.getElementById("textbox_latentend").value = document.getElementById("textbox_step").value;
    OnStepTextChanged();
    OnLatentEndTextChanged();
    
    if(document.getElementById("checkbox_center").checked)
        CenterTarget();
}

function OnDrawAll()
{
    draw_all = true;
    DrawCanvas();
    for(var k = 0; k < vehicles.length; ++k)
    {
        RequestOutput(k);

    }
}