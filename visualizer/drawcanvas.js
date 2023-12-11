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
var draw_all = false;

var current_step = 0;
var vehicles = [];
var predicteds = [[]];
var latent_predicted_mu = null;
var latent_predicted_var = null;
var latent_data = {};
var latent_output = {};
var latent_output_prob = {};
var latent_idx = null;
var real_output = {};

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
    draw = []
    if (draw_all)
    {
        for(var k = 0; k < vehicles.length; ++k)
            draw.push(k)
    }
    else if(clicked != -1)
    {
        draw.push(clicked)
    }
    for(c of draw)
    {
        drawctx.strokeStyle = "rgba(0, 255, 0)";
        drawctx.lineWidth = 0.5;
        if(c in real_output)
        {
            if(real_output[c].length > 1)
            {
                drawctx.beginPath();
                drawctx.moveTo(real_output[c][0][0], real_output[c][0][1]);
                for(var i = 1; i < real_output[c].length; ++i)
                {
                    drawctx.lineTo(real_output[c][i][0], real_output[c][i][1]);
                }
                drawctx.stroke();
    
            }
        }
        if(c in latent_output)
        {
            if(latent_output[c].length > 1)
            {
                v = vehicles[c];
    
                drawctx.save()
                drawctx.transform(1, 0, 0, 1, v[0], v[1])
                drawctx.rotate(v[2])
                for(var action = 0; action < latent_output[c].length; action++)
                {
                    l = latent_output[c][action]
                    prob = Math.sqrt(latent_output_prob[c][action])
    
                    drawctx.strokeStyle = "rgba(255, 0, 0, " + prob + ")";
                    drawctx.lineWidth = 0.1;
    
                    drawctx.beginPath();
                    drawctx.moveTo(0, 0);
                    for(var i = 0; i < l.length / 2; ++i)
                    {
                        drawctx.lineTo(l[i * 2], l[i * 2 + 1]);
                    }
                    drawctx.stroke();
                    
    
                    
                }
                
                
    
                drawctx.restore()
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
        drawall = false;
        //document.getElementById("slider_impatience").value = impatiences[clicked]
        if(clicked in latent_data)
        {
            linechart.data.datasets = latent_data[clicked];
            linechart.update('none');
            RequestLatentPredicted(clicked);
        }
        else
        {
            RequestLatentData(clicked);
        }
    }
    draw_all = false;
    DrawCanvas();
}

function CenterTarget()
{
    if(clicked != -1)
    {

        viewport_x = (vehicles[clicked][1] * carla_scale - carla_x) * viewport_scale + 320
        viewport_y = 320  - (vehicles[clicked][0] * carla_scale + carla_y) * viewport_scale
    }

}