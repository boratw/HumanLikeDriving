let visctx;
let drawctx;
let laneimage;

let viewport_x = 0;
let viewport_y = 0;
let viewport_scale = 1;
let viewport_screen_ratio = 1.0;
let half_viewport_width = 320;
let half_viewport_height = 320;

const carla_scale = 5.44;
const carla_x = 1365;
const carla_y = 1350;
const carla_rotate = 1.570796327;

var clicked = -1;

var current_step = 0;
var vehicles = [];
var latents = {};
var predicteds = [[]];
var latent_data = {};
var latent_output = undefined;
var latent_idx = 10;
var real_output = undefined;

let draw_potential = false;

function InitCanvas()
{

    visctx = document.getElementById("canvas").getContext("2d");
    drawctx = document.createElement("canvas").getContext("2d");
    linectx = document.createElement("canvas").getContext("2d");
    laneimage = document.getElementById("world_background");

    viewport_screen_ratio = visctx.canvas.width / visctx.canvas.clientWidth;
    half_viewport_width = visctx.canvas.clientWidth / 2;
    half_viewport_height = visctx.canvas.clientHeight / 2;

    drawctx.canvas.width = visctx.canvas.width;
    drawctx.canvas.height = visctx.canvas.height;
    linectx.canvas.width = visctx.canvas.width;
    linectx.canvas.height = visctx.canvas.height;
}

function DrawCanvas()
{
    drawctx.resetTransform()
    drawctx.fillStyle = "rgb(0, 0, 0)";
    drawctx.fillRect(0, 0, 640, 640);

    drawctx.setTransform(viewport_scale, 0, 0, viewport_scale, viewport_x, viewport_y);
    drawctx.drawImage(laneimage, 0, 0)


    drawctx.transform(carla_scale, 0, 0, carla_scale, carla_x, carla_y)
    drawctx.rotate(carla_rotate)
    drawctx.fillStyle = "rgb(0, 255, 0)";
    drawctx.strokeStyle = "rgba(0, 255, 0, 0.5)";
    drawctx.lineWidth = 0.5;

    for(var k = 0; k < vehicles.length; ++k)
    {
        v = vehicles[k];
        drawctx.save()
        if(k == clicked)
            drawctx.fillStyle = "rgb(255, 255, 0)";
        drawctx.transform(1, 0, 0, 1, v[0], v[1])
        drawctx.rotate(v[2])
        drawctx.fillRect(-1.8, -0.7, 3.6, 1.4);
        drawctx.restore()

    }

    if(real_output != undefined)
    {
        if(real_output.length > 1)
        {
            drawctx.beginPath();
            drawctx.moveTo(real_output[0][0], real_output[0][1]);
            for(var i = 1; i < real_output.length; ++i)
            {
                drawctx.lineTo(real_output[i][0], real_output[i][1]);
            }
            drawctx.stroke();

        }
    }
    drawctx.strokeStyle = "rgba(255, 0, 0, 0.5)";
    if(latent_output != undefined)
    {
        if(latent_output.length > 1)
        {
            if(latent_idx == null)
            {
                for(v of latent_output)
                {
                    drawctx.beginPath();
                    drawctx.moveTo(v[0][0], v[0][1]);
                    for(var i = 1; i < v.length; ++i)
                    {
                        drawctx.lineTo(v[i][0], v[i][1]);
                    }
                    drawctx.stroke();
    
                }
            }
            else
            {
                v = latent_output[latent_idx]
                drawctx.beginPath();
                drawctx.moveTo(v[0][0], v[0][1]);
                for(var i = 1; i < v.length; ++i)
                {
                    drawctx.lineTo(v[i][0], v[i][1]);
                }
                drawctx.stroke();

            }
        }
    }

    visctx.drawImage(drawctx.canvas, 0, 0);
}

function CanvasClick(x, y)
{
    x = ((x  - viewport_x) / viewport_scale - carla_x) / carla_scale
    y = ((y  - viewport_y) / viewport_scale - carla_y) / carla_scale
    fx = y
    fy = -x
    min = 10;
    minidx = -1;
    for(var k = 0; k < vehicles.length; ++k)
    {
        d = Math.abs(vehicles[k][0] - fx) + Math.abs(vehicles[k][1] - fy)
        if(d < min)
        {
            min = d;
            minidx = k;
        }
    }
    clicked = minidx;
    latentoutput = undefined;

    if(clicked != -1)
    {
        if(clicked in latent_data)
        {
            linechart.data.datasets = latent_data[clicked];
            linechart.update('none');
            MoveLatentToDefault();
        }
        else
        {
            RequestCurrentAgent(clicked);
        }
    }
    DrawCanvas();
}

function MoveLatentToDefault()
{
    for(var i = 0; i < latent_length; ++i)
    {
        box = document.getElementById("box_l" + i)
        slider = document.getElementById("slider_l" + i)
        //mu = Math.round(latents[clicked][i][0] * 100)
        //l = mu * 0.45 + 180 - latents[clicked][i][1] * 45
        //r = mu * 0.45 + 180 + latents[clicked][i][1] * 45
        mu = Math.round(latents[clicked][i] * 100)
        l = mu * 1.8 + 180 - 0.1 * 45
        r = mu * 1.8 + 180 + 0.1 * 45
        if(l < 0)
            l = 0
        else if (l > 360)
            l = 360
        if(r < 0)
            r = 0
        else if (r > 360)
            r = 360
        box.style.left = l + "px"
        box.style.width = (r-l) + "px"
        document.getElementById("slider_l" + i).value = mu
        document.getElementById("value_l" + i).innerHTML = mu / 100
    }


    RequestLatentResult();

}