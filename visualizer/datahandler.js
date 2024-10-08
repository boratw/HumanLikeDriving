function RequestCurrentMap()
{
    var xmlHttp2 = new XMLHttpRequest();
    xmlHttp2.onreadystatechange = function() {
        if(this.status == 200 && this.readyState == this.DONE) {
            HandleCurrentExp(xmlHttp2.responseText);            
        }
    };
    
    xmlHttp2.open("GET", "/v/expinfo/", false);
    xmlHttp2.send();
}

function RequestCurrentStep()
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if(this.status == 200 && this.readyState == this.DONE) {
            HandleCurrentStep(xmlHttp.responseText);            
        }
    };
    
    xmlHttp.open("GET", "/v/curstate/" + current_step, false);
    xmlHttp.send();
    if(draw_all)
    {
        for(var k = 0; k < vehicles.length; ++k)
        {
            RequestOutput(k, false);
        }

    }
    else if(clicked != -1)
        RequestOutput(clicked, false);
    DrawCanvas();

}
function RequestOutput(target, draw=true)
{
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.onreadystatechange = function() {
            if(this.status == 200 && this.readyState == this.DONE) {
                HandleOutput(target, xmlHttp.responseText);  
                if(draw)
                    DrawCanvas()          
            }
        };
        url = "/v/predictroute/" + target + "/";
        if(latent_changed)
        {
            for(var i = 0; i < latent_length; ++i)
                url += document.getElementById("value_l" + i).innerHTML + "/"
        }
        else
        {
            url += document.getElementById("textbox_latentstart").value + "/" + document.getElementById("textbox_latentend").value  + "/";
        }
        xmlHttp.open("GET", url, false);
        xmlHttp.send();
}
function RequestLatentData(target)
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if(this.status == 200 && this.readyState == this.DONE) {
            HandleLatentData(target, xmlHttp.responseText);            
        }
    };
    url = "/v/latents/" + target + "/";
    xmlHttp.open("GET", url, false);
    xmlHttp.send();
}
function RequestLatentPredicted(target)
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if(this.status == 200 && this.readyState == this.DONE) {
            HandleLatentPredicted(target, xmlHttp.responseText);            
        }
    };
    url = "/v/predictlatent/" + target + "/" + predictor_index + "/" + document.getElementById("textbox_latentstart").value + "/" + document.getElementById("textbox_latentend").value  + "/";
    xmlHttp.open("GET", url, false);
    xmlHttp.send();
}
function RequestExp(idx, exp)
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() {
        if(this.status == 200 && this.readyState == this.DONE) {
            RequestCurrentStep();            
        }
    };
    url = "/v/setexp/" + idx + "/" + exp + "/";
    xmlHttp.open("GET", url, false);
    xmlHttp.send();
}

function HandleCurrentStep(response)
{
    data = JSON.parse(response);
    vehicles = data["state"];
}

function HandleCurrentExp(response)
{
    data = JSON.parse(response);
    max_step = data["max_step"];
    latent_length = data["latent_len"];

    document.getElementById("slider_step").max = max_step;
    document.getElementById("slider_latentstart").max = max_step;
    document.getElementById("slider_latentend").max = max_step;
    for(var i = 0; i < 8; ++i)
    {
        if(i >= latent_length)
            document.getElementById("div_l" + i).hidden = true;
    }
    
    labels =[]
    for(var i = 0; i < max_step; ++i)
    {
        labels.push(i)
    }

    linechart.data.labels = labels;
    

}
function HandleOutput(target, response)
{
    data = JSON.parse(response);
    real_output[target] = data["route"];
    latent_output[target] = data["predicted"];
    latent_output_prob[target] = data["action_prob"];
    latent_used[target] = data["latent"];
    if("mask" in data)
    {
        mask_used[target] = data["mask"];
    
        for(var i = 0; i < data["mask"].length; ++i)
        {
            c = Math.floor(data["mask"][i] * 128)
            document.getElementById("div_mask_" + i).style.backgroundColor = `rgb(${c}, ${c}, ${c})`
        }
        
    }
    if("zero_predicted" in data)
    {
        zero_latent_output[target] = data["zero_predicted"];
    }
}
function HandleLatentData(c, response)
{
    
    j = JSON.parse(response);
    datalist = [];
    const bordercolors = ["#1e90ff", "#ff1493", "#228b22", "#daa520", "#4B0082", "#4169E1", "#008B8B", "#006400"]
    for(var i = 0; i < latent_length; ++i)
    {
        datalist.push({
            label: "global" + i,
            data: [],
            borderColor: bordercolors[i],
            fill: false})
    }
    for(var i = 0; i < j["mu"].length; ++i)
    {
        datalist[0]["data"].push(j["mu"][i][0]);
        datalist[1]["data"].push(j["mu"][i][1]);
        datalist[2]["data"].push(j["mu"][i][2]);
        datalist[3]["data"].push(j["mu"][i][3]);
    }
    latent_data[c] = datalist;
    if(clicked == c)
    {
        linechart.data.datasets = datalist;
        linechart.update('none');

        RequestLatentPredicted(c);
    }
    
    RequestOutput(c);
}

function HandleLatentPredicted(c, response)
{
     
    if(clicked == c)
    {
        data = JSON.parse(response);
        latent_predicted_mu = data["mu"];
        latent_predicted_var = data["std"];
        for(var i = 0; i < latent_length; ++i)
        {
            box = document.getElementById("box_l" + i)
            slider = document.getElementById("slider_l" + i)

            mu = Math.round(latent_predicted_mu[i] * 100)
            //l = mu * 0.45 + 180 - Math.max(latent_predicted_var[i], 0.1) * 45
            //r = mu * 0.45 + 180 + Math.max(latent_predicted_var[i], 0.1) * 45
            l = mu * 0.45 + 170
            r = mu * 0.45 + 190

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
        RequestOutput(c);
    }
    RequestOutput(c);

}
