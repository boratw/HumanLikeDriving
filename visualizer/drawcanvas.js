let visctx;
let drawctx;
let laneimage;

let viewport_x = 0;
let viewport_y = 0;
let viewport_scale = 1;
let viewport_screen_ratio = 1.0;
let half_viewport_width = 320;
let half_viewport_height = 320;

const carla_scale = 10.55;
const carla_x = 1410;
const carla_y = 1040;
const carla_rotate = 0;

var clicked = -1;

var current_step = 0;
var vehicles = [];
var routes = [[]];
var latents = undefined;
var predicteds = [[]];
var latent_data = {};
var latentoutput = undefined;

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
    linectx.resetTransform()
    linectx.fillStyle = "rgb(0, 0, 0)";
    linectx.fillRect(0, 0, 640, 640);

    linectx.setTransform(viewport_scale, 0, 0, viewport_scale, viewport_x, viewport_y);
    linectx.drawImage(laneimage, 0, 0)

    linectx.transform(carla_scale, 0, 0, carla_scale, carla_x, carla_y)
    linectx.rotate(carla_rotate)
    linectx.lineCap = "round";
    linectx.lineJoin = "round";

    
    if(draw_potential)
    {
        linectx.filter = "blur(10px)";
        linectx.strokeStyle = "rgba(255, 0, 0, 0.2)";
        linectx.lineWidth = 3;
    }
    else
    {
        linectx.strokeStyle = "rgba(255, 0, 0, 0.2)";
        linectx.filter = "none";
        linectx.lineWidth = 1;

    }
    /*
    if(latentoutput != undefined)
    {
        linectx.strokeStyle = "rgba(255, 0, 0, 1)";
        linectx.filter = "none";
        linectx.lineWidth = 0.25;
        linectx.beginPath();
        if(latentoutput.length > 1)
        {
            linectx.moveTo(latentoutput[0][0], latentoutput[0][1]);
            for(var i = 1; i < v.length; ++i)
            {
                linectx.lineTo(latentoutput[i][0], latentoutput[i][1]);
            }

        }
        linectx.stroke();
    }
    else
    {
        if(latentoutput == undefined)
        {
            for(v of predicteds[clicked])
            {
                linectx.beginPath();
                if(v.length > 1)
                {
                    linectx.moveTo(v[0][0], v[0][1]);
                    for(var i = 1; i < v.length; ++i)
                    {
                        linectx.lineTo(v[i][0], v[i][1]);
                    }
        
                }
                linectx.stroke();
            }
        }
        else
        {
            linectx.strokeStyle = "rgba(255, 0, 0, 1)";
            linectx.beginPath();
            if(latentoutput.length > 1)
            {
                linectx.moveTo(latentoutput[0][0], latentoutput[0][1]);
                for(var i = 1; i < v.length; ++i)
                {
                    linectx.lineTo(latentoutput[i][0], latentoutput[i][1]);
                }
    
            }
            linectx.stroke();

        }
    }
    */
    if(clicked == -1)
    {
        for(var k = 0; k < predicteds.length; ++k)
        {
            for(v of predicteds[k])
            {
                linectx.beginPath();
                if(v.length > 1)
                {
                    linectx.moveTo(v[0][0], v[0][1]);
                    for(var i = 1; i < v.length; ++i)
                    {
                        linectx.lineTo(v[i][0], v[i][1]);
                    }
        
                }
                linectx.stroke();

            }
        }
    }
    else
    {
        if(latentoutput == undefined)
        {
            for(v of predicteds[clicked])
            {
                linectx.beginPath();
                if(v.length > 1)
                {
                    linectx.moveTo(v[0][0], v[0][1]);
                    for(var i = 1; i < v.length; ++i)
                    {
                        linectx.lineTo(v[i][0], v[i][1]);
                    }
        
                }
                linectx.stroke();
            }
        }
        else
        {
            for(v of latentoutput)
            {
                linectx.beginPath();
                if(v.length > 1)
                {
                    linectx.moveTo(v[0][0], v[0][1]);
                    for(var i = 1; i < v.length; ++i)
                    {
                        linectx.lineTo(v[i][0], v[i][1]);
                    }
        
                }
                linectx.stroke();
            }

        }
    }
    

    drawctx.resetTransform()
    drawctx.clearRect(0, 0, 640, 640);

    drawctx.setTransform(viewport_scale, 0, 0, viewport_scale, viewport_x, viewport_y);

    drawctx.transform(carla_scale, 0, 0, carla_scale, carla_x, carla_y)
    drawctx.rotate(carla_rotate)
    drawctx.fillStyle = "rgb(0, 255, 0)";
    drawctx.strokeStyle = "rgba(0, 255, 0, 0.5)";
    drawctx.lineWidth = 0.25;

    drawctx.beginPath();
    for(var k = 0; k < routes.length; ++k)
    {
        v = routes[k];
        if(v.length > 1)
        {
            drawctx.moveTo(v[0][0], v[0][1]);
            for(var i = 1; i < v.length; ++i)
            {
                drawctx.lineTo(v[i][0], v[i][1]);
            }

        }
    }
    drawctx.stroke();

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

    visctx.drawImage(linectx.canvas, 0, 0);
    visctx.drawImage(drawctx.canvas, 0, 0);

    /*
    if(clicked != -1)
    {
        if(latentoutput == undefined)
        {
            for(var i = 0; i < 8; ++i)
            {
                box = document.getElementById("box_l" + i)
                slider = document.getElementById("slider_l" + i)
                mu = Math.round(latents[clicked][i][0] * 100)
                l = mu * 0.45 + 180 - latents[clicked][i][1] * 45
                r = mu * 0.45 + 180 + latents[clicked][i][1] * 45
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

        }

    }
    */
}

function DrawSliders()
{
    if(clicked != -1 && latents != undefined)
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

    }

}


function CanvasClick(x, y)
{
    x = ((x  - viewport_x) / viewport_scale - carla_x) / carla_scale
    y = ((y  - viewport_y) / viewport_scale - carla_y) / carla_scale
    min = 10;
    minidx = -1;
    for(var k = 0; k < vehicles.length; ++k)
    {
        d = Math.abs(vehicles[k][0] - x) + Math.abs(vehicles[k][1] - y)
        if(d < min)
        {
            min = d;
            minidx = k;
        }
    }
    clicked = minidx;
    latentoutput = undefined;
    DrawSliders();
    DrawCanvas();

    if(clicked != -1)
    {
        if(clicked in latent_data)
        {
            linechart.data.datasets = latent_data[clicked];
            linechart.update('none');
        }
        else
        {
            RequestCurrentAgent();
        }
    }
}